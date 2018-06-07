#include <string.h>
#include <msp430.h>
#include <libio/console.h>
#include <libmspdriver/driverlib.h>
#include <libdsp/DSPLib.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "blas.h"
#include "mem.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"

static __fram mat_t m1 = {.data = MAT_BUFFER(0)};
static __fram mat_t m2 = {.data = MAT_BUFFER(1)};
static __fram mat_t *inter1 = &m1;
static __fram mat_t *inter2 = &m2;
static DSPLIB_DATA(tsrc1, 2) fixed tsrc1[CONFIG_TILE_SIZE];
static DSPLIB_DATA(tsrc2, 2) fixed tsrc2[CONFIG_TILE_SIZE];
static DSPLIB_DATA(tsrc3, 2) fixed tsrc3[MAT_BUFFER_SIZE];
static DSPLIB_DATA(tsrc4, 2) fixed tsrc4[MAT_BUFFER_SIZE];
static DSPLIB_DATA(tdest, 2) fixed tdest[2];

static __fram DMA_initParam dmaConfig[3];
static uint DMA_initialized = 0;

static __fram uint scratch_bak[SCRATCH_SIZE];
static __fram uint tile_size;

// Public tasks
TASK(TASK_UID_BLAS_OFFSET, task_ds_zero);
TASK(TASK_UID_BLAS_OFFSET + 1, task_ds_add);
TASK(TASK_UID_BLAS_OFFSET + 2, task_ds_mul);
TASK(TASK_UID_BLAS_OFFSET + 3, task_ds_div);
TASK(TASK_UID_BLAS_OFFSET + 4, task_dm_add);
TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_mul);
TASK(TASK_UID_BLAS_OFFSET + 6, task_dm_conv);
TASK(TASK_UID_BLAS_OFFSET + 7, task_dm_conv_same);
TASK(TASK_UID_BLAS_OFFSET + 8, task_sm_mul);
TASK(TASK_UID_BLAS_OFFSET + 9, task_sm_conv);
TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv_same);

// Private tasks
void task_cleanup_blas();
TASK(TASK_UID_BLAS_OFFSET + 11, task_cleanup_blas);

void task_calibrate();
TASK(TASK_UID_BLAS_OFFSET + 12, task_calibrate);

void task_dm_mul_flex();
void task_dm_mul_lea();
void task_sm_conv_flex();
void task_sm_conv_lea();
TASK(TASK_UID_BLAS_OFFSET + 13, task_dm_mul_flex);
TASK(TASK_UID_BLAS_OFFSET + 14, task_dm_mul_lea);
TASK(TASK_UID_BLAS_OFFSET + 15, task_sm_conv_flex);
TASK(TASK_UID_BLAS_OFFSET + 16, task_sm_conv_lea);

void pulse(uint pin) {
	P6DIR = 0x03;
	P6OUT = pin;
	__delay_cycles(0x100);
	P6OUT = 0x00;
}

// Shut off CPU and do a DMA transfer
void DMA_startSleepTransfer(uint channel) {
	uint16_t interruptState = __get_interrupt_state();
    __disable_interrupt();
   	DMA_startTransfer(channel);
	__bis_SR_register(GIE + LPM0_bits); 
	__set_interrupt_state(interruptState);	
}

uint greatest_tile_size(uint dim, uint max) {
	if(dim < max) return dim;

	uint i = 2;
	uint max_divisor = i;
	while(i <= max && i <= dim) {
        if(dim % i == 0) max_divisor = i;
    	i += 2;
    }
    return max_divisor;
}

#define LEA_WORK_CUTOFF 2
void check_calibrate(task_t *flex, task_t *lea) {
	task_t *target = CUR_TASK;
	if(flex != NULL && lea != NULL) {
		target = flex;
		mat_t *filter = PEEK_STACK(mat_stack, 2);
		uint fitems = MAT_GET_DIM(filter, 2);
		if(filter->len_dims == 2) {
			fitems = MAT_GET_DIM(filter, 1);
		} else if(filter->sparse.len_dims != 0) {
			fitems = MAT_GET_DIM(filter, 0);
		}
		if(fitems >= LEA_WORK_CUTOFF) {
			PRINTF("\r\n Choosing LEA %u", fitems);
			target = lea;
		}
		target->info.return_task = CUR_INFO.return_task;
		if(tile_size != 0) transition_to(target);
	}

	if(!DMA_initialized) {
		PRINTF("\r\n Initializing DMA");
		DMA_disableTransferDuringReadModifyWrite();
		for(uint i = 0; i < 3; i++) {
			dmaConfig[i].channelSelect = i << 4;
			dmaConfig[i].transferModeSelect = DMA_TRANSFER_BLOCK;
			dmaConfig[i].transferUnitSelect = DMA_SIZE_SRCWORD_DSTWORD;
			DMA_init(&dmaConfig[i]);
			DMA_enableInterrupt(dmaConfig[i].channelSelect);
		}
		DMA_initialized = 1;
	}

	if(tile_size != 0) return;
	TASK_REF(task_calibrate)->info.return_task = target;
	TRANSITION_TO(task_calibrate);	
}

// Resets a task
static __fram task_t *last_task;
void task_cleanup_blas() {
	// PRINTF("\r\n     Finishing BLAS");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

void task_calibrate() {
	PRINTF("\r\n Calibrating...");
	if(CUR_INFO.scratch[0] == 0) {
		CUR_INFO.scratch[1] = CONFIG_TILE_SIZE;
		CUR_INFO.scratch[0] = 1;
#ifdef CONFIG_INTERMITTENT
		while(1) {}
#else
		transition_to(CUR_TASK);
#endif
	}

	if(CUR_INFO.scratch[0] == 3) {
		CUR_INFO.scratch[0] = 1;
#ifdef CONFIG_INTERMITTENT
		while(1) {}
#else
		transition_to(CUR_TASK);
#endif
	} else if(CUR_INFO.scratch[0] == 1 && tile_size == 0) {
		CUR_INFO.scratch[0] = 2;
		msp_mac_q15_params params = {.length = CUR_INFO.scratch[1]};
		msp_status status;
		status = msp_mac_q15(&params, tsrc1, tsrc2, tdest);
		PRINTF("\r\n Done init: status: %u tile_size %u", status, CUR_INFO.scratch[1]);
		msp_checkStatus(status);
		write_to_gbuf((uint8_t *)(CUR_INFO.scratch + 1), (uint8_t *)(&tile_size), sizeof(uint));
		transition_to(CUR_TASK);
	} else if(CUR_INFO.scratch[0] == 2 && tile_size == 0) {
		scratch_bak[0] = 3;
		scratch_bak[1] = CUR_INFO.scratch[1] / 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		transition_to(CUR_TASK);	
	}
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense scalar addition
void task_ds_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense scalar multiplication
void task_ds_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			fixed w = F_MUL(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense scalar division
void task_ds_div() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			fixed w = F_DIV(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense set mat to all 0s
void task_ds_zero() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			MAT_SET(dest, 0, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 2);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix addition
void task_dm_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, i, j));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix multiplication
void task_dm_mul() {
	check_calibrate(TASK_REF(task_dm_mul_flex), TASK_REF(task_dm_mul_lea));
}

void task_dm_mul_flex() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(filter, 0);
	uint cols = MAT_GET_DIM(filter, 1);
	uint dcols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter1, rows, dcols);
	MAT_RESHAPE(inter2, rows, dcols);

	uint k = CUR_INFO.scratch[2];
	mat_t *prev_dest = (k % 2 == 0) ? inter2 : inter1;
	if(k < cols - 1) {
		dest = (k % 2 == 0) ? inter1 : inter2;
	}

	if(k > 0) {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < dcols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, i, k), MAT_GET(src, k, j));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < dcols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, i, k), MAT_GET(src, k, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k + 1;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	if(k < cols - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_dm_mul_lea() {
	check_calibrate(NULL, NULL);
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(filter, 0);
	uint cols = MAT_GET_DIM(filter, 1);
	uint dcols = MAT_GET_DIM(dest, 1);
	uint common_tile_size = greatest_tile_size(cols, tile_size);
	if(common_tile_size & 0x01) {
		common_tile_size -= 1;
	}

	uint zero = CUR_INFO.scratch[0];
	if(zero == 0) {
		if(MAT_GET_DIM(src, 1) == 1) {
			PRINTF("\r\n Quick transpose");
			MAT_RESHAPE(src, 1, MAT_GET_DIM(src, 0));
		} else { // transpose Here

		}
		scratch_bak[0] = 1;
		scratch_bak[1] = ((cols / common_tile_size) % 2)? 0 : 1; // Buffering
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(CUR_INFO.scratch[1]) { // Swap
		dest = inter;
		inter = tmp;
	}

	msp_mac_q15_params params = {.length = common_tile_size};
	uint j = CUR_INFO.scratch[2];
	PRINTF("\r\n j: %u tile_size: %u swapped: %u", j, common_tile_size, CUR_INFO.scratch[1]);
	for(uint i = CUR_INFO.scratch[3]; i < rows; i = ++CUR_INFO.scratch[3]) {
		// DMA Filter
		DMA_setTransferSize(dmaConfig[0].channelSelect, common_tile_size);
	    DMA_setSrcAddress(dmaConfig[0].channelSelect,
	                      (uint32_t) MAT_PTR(filter, i, j),
	                      DMA_DIRECTION_INCREMENT);

	    DMA_setDstAddress(dmaConfig[0].channelSelect,
	                      (uint32_t) (tsrc1),
	                      DMA_DIRECTION_INCREMENT);
	    DMA_enableTransfers(dmaConfig[0].channelSelect);
	    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
		for(uint k = CUR_INFO.scratch[4]; k < dcols; k = ++CUR_INFO.scratch[4]) {
			// DMA Src
			DMA_setTransferSize(dmaConfig[1].channelSelect, common_tile_size);
		    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
							(uint32_t) MAT_PTR(src, k, j),
                      		DMA_DIRECTION_INCREMENT);
		    DMA_setDstAddress(dmaConfig[1].channelSelect,
		                    (uint32_t) (tsrc2),
		                    DMA_DIRECTION_INCREMENT);
			DMA_enableTransfers(dmaConfig[1].channelSelect);
		    DMA_startSleepTransfer(dmaConfig[1].channelSelect);	
			fixed w = 0;
			if(zero == 2) w = MAT_GET(inter, i, k);
			msp_mac_q15(&params, tsrc1, tsrc2, tdest);
			fixed rounded = ((*tdest >> 1) + F_K) >> F_N;
    		w = F_ADD(w, rounded);
    		MAT_SET(dest, w, i, k);	
		}
		CUR_INFO.scratch[4] = 0;
	}
	// Increment by common_tile_size
	scratch_bak[0] = 2; // zero
	scratch_bak[1] = CUR_INFO.scratch[1] ^ 0x01; // swap
	scratch_bak[2] = j + common_tile_size; // j
	scratch_bak[3] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	if(j + common_tile_size < cols) transition_to(CUR_TASK);
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution
void task_dm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint k = CUR_INFO.scratch[2];
	uint l = CUR_INFO.scratch[3];
	uint n = CUR_INFO.scratch[4];
	mat_t *prev_dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k;
	scratch_bak[3] = l;
	if(n + 1 == fcols && l + 1 == frows) {
		scratch_bak[3] = 0;
		scratch_bak[2] = k + 1;
	} else if(n + 1 == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + 1 == fcols) ? 0 : n + 1;
	scratch_bak[5] = ~CUR_INFO.scratch[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution
void task_dm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint k = CUR_INFO.scratch[2];
	uint l = CUR_INFO.scratch[3];
	uint n = CUR_INFO.scratch[4];
	mat_t *prev_dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
					w = 0;
				}
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k;
	scratch_bak[3] = l;
	if(n + 1 == fcols && l + 1 == frows) {
		scratch_bak[3] = 0;
		scratch_bak[2] = k + 1;
	} else if(n + 1 == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + 1 == fcols) ? 0 : n + 1;
	scratch_bak[5] = ~CUR_INFO.scratch[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Sparse matrix multiplication
void task_sm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0); // n => i
	uint cols = MAT_GET_DIM(src, 0); // m => k
	uint dcols = MAT_GET_DIM(dest, 1); // p => j
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, dcols);

	uint pos = CUR_INFO.scratch[0];
	uint i = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];
	char zero = CUR_INFO.scratch[3];

	if(zero == 0) {
		scratch_bak[2] = filter->sparse.offsets[pos];
		scratch_bak[1] = scratch_bak[2] / cols;
		scratch_bak[2] %= cols;
		scratch_bak[3] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(total_elements % 2 == 0 && pos % 2 == 0) { // A
		dest = inter;
		inter = tmp;
	} else if(total_elements % 2 == 1 && pos % 2 == 1) { // B
		dest = inter;
		inter = tmp;
	}

	for(uint j = CUR_INFO.scratch[4]; j < dcols; j = ++CUR_INFO.scratch[4]) {
		fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
		if(zero == 2) {
			w = F_ADD(w, MAT_GET(inter, i, j));
		}
		MAT_SET(dest, w, i, j);
		write_to_gbuf((uint8_t *)(dest->data + i * dcols + j), (uint8_t *)(inter->data + i * dcols + j), sizeof(uint));
	}

	scratch_bak[0] = pos + 1;
	scratch_bak[2] = k + filter->sparse.offsets[pos + 1];
	scratch_bak[3] = (scratch_bak[2] / cols > 0) ? 1 : 2;
	scratch_bak[1] = i + scratch_bak[2] / cols;
	scratch_bak[2] %= cols;
	scratch_bak[4] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv() {
	check_calibrate(TASK_REF(task_sm_conv_flex), TASK_REF(task_sm_conv_lea));
}

void task_sm_conv_flex() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(total_elements % 2 == 0 && pos % 2 == 0) { // A
		dest = inter;
		inter = tmp;
	} else if(total_elements % 2 == 1 && pos % 2 == 1) { // B
		dest = inter;
		inter = tmp;
	}

	uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	uint n = idx % fcols; // Cols
	for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
		for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
			fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n) >> SHIFT);
			if(zero == 2) {
				w = F_ADD(w, MAT_GET(inter, i, j));
			}
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[3] = 0;
	}
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

__fram fixed coalesced_filter[CONFIG_TILE_SIZE];
void task_sm_conv_lea() {
	check_calibrate(NULL, NULL);
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint slayers = MAT_GET_DIM(src, 0); // So metal
	uint srows = MAT_GET_DIM(src, 1);
	uint scols = MAT_GET_DIM(src, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 0);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);

	uint common_tile_size = greatest_tile_size(scols, tile_size);
	uint even_offset = common_tile_size % 2;
	uint feven_offset = (fcols % 2) ^ 0x01;
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	char state = CUR_INFO.scratch[4];

	if(state == 0) { // Generate Initial Filter
		memset(coalesced_filter, 0, sizeof(filter) * (fcols + feven_offset));
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[1] = pos;
		uint offset = scratch_bak[0] % fcols;
		uint init_row = scratch_bak[0] / fcols;
		coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
		for(uint i = 0; i < fcols; i++) {
			if((scratch_bak[0] + filter->sparse.offsets[scratch_bak[1] + 1]) / fcols != init_row || 
				scratch_bak[1] + 1 >= total_elements) break;

			scratch_bak[1]++;
			scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
			offset = scratch_bak[0] % fcols;
			coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
		}
		scratch_bak[4] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(CUR_INFO.scratch[5]) { // Swap
		dest = inter;
		inter = tmp;
	}

    uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	// PRINTF("\r\n tile_size: %u length: %u tapLength: %u k: %u l: %u", 
		// common_tile_size, common_tile_size + even_offset, fcols + fcols % 2, k, l);

	msp_status status;
	msp_fir_q15_params params_fir = {.length = common_tile_size + even_offset, 
									.tapLength = (fcols + fcols % 2), .coeffs = tsrc1};
	msp_add_q15_params params_add = {.length = common_tile_size + even_offset};
	uint fsize = fcols + fcols % 2;
	if(fsize > 12) {
		DMA_setTransferSize(dmaConfig[0].channelSelect, fsize);
	    DMA_setSrcAddress(dmaConfig[0].channelSelect,
	                      (uint32_t) (coalesced_filter),
	                      DMA_DIRECTION_INCREMENT);

	    DMA_setDstAddress(dmaConfig[0].channelSelect,
	                      (uint32_t) (tsrc1),
	                      DMA_DIRECTION_INCREMENT);
	    DMA_enableTransfers(dmaConfig[0].channelSelect);
	    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
	} else {
		memcpy(tsrc1, coalesced_filter, sizeof(fixed) * (fcols + fcols % 2));
	}

	if(common_tile_size > 12) {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			for(uint j = CUR_INFO.scratch[3]; j < scols; j = (CUR_INFO.scratch[3] += common_tile_size)) {
				DMA_setTransferSize(dmaConfig[0].channelSelect, common_tile_size);
			    DMA_setSrcAddress(dmaConfig[0].channelSelect,
			                      (uint32_t) (MAT_PTR(src, k, i + l, j)),
			                      DMA_DIRECTION_INCREMENT);

			    DMA_setDstAddress(dmaConfig[0].channelSelect,
			                      (uint32_t) (tsrc3),
			                      DMA_DIRECTION_INCREMENT);
			    DMA_enableTransfers(dmaConfig[0].channelSelect);
			    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
				if(even_offset) tsrc3[common_tile_size + even_offset] = 0;
				status = msp_fir_q15(&params_fir, tsrc3, tsrc2);
				if(state == 3) {
					DMA_setTransferSize(dmaConfig[0].channelSelect, common_tile_size);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(inter, i, j)),
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (tsrc4),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
					if(even_offset) tsrc4[common_tile_size + even_offset] = 0;
					status = msp_add_q15(&params_add, tsrc2, tsrc4, tsrc3);
					DMA_setTransferSize(dmaConfig[0].channelSelect, common_tile_size);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) tsrc3,
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(dest, i, j)),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
				} else {
					DMA_setTransferSize(dmaConfig[0].channelSelect, common_tile_size);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) tsrc2,
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(dest, i, j)),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
				}
			}
			CUR_INFO.scratch[3] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			for(uint j = CUR_INFO.scratch[3]; j < scols; j = (CUR_INFO.scratch[3] += common_tile_size)) {
				memcpy(tsrc3, MAT_PTR(src, k, i + l, j), sizeof(fixed) * common_tile_size);
				if(even_offset) tsrc3[common_tile_size + even_offset] = 0;
				status = msp_fir_q15(&params_fir, tsrc3, tsrc2);
				if(state == 3) {
					memcpy(tsrc4, MAT_PTR(inter, i, j), sizeof(fixed) * common_tile_size);
					if(even_offset) tsrc4[common_tile_size + even_offset] = 0;
					status = msp_add_q15(&params_add, tsrc2, tsrc4, tsrc3);
					memcpy(MAT_PTR(dest, i, j), tsrc3, sizeof(fixed) * common_tile_size);
				} else {
					memcpy(MAT_PTR(dest, i, j), tsrc2, sizeof(fixed) * common_tile_size);
				}
			}
			CUR_INFO.scratch[3] = 0;
		}
	}

	// Coalesced Filter
	memset(coalesced_filter, 0, sizeof(filter) * fsize);
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	uint offset = scratch_bak[0] % fcols;
	uint init_row = scratch_bak[0] / fcols;
	coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
	for(uint i = 0; i < fcols; i++) {
		if((scratch_bak[0] + filter->sparse.offsets[scratch_bak[1] + 1]) / fcols != init_row || 
			scratch_bak[1] + 1 >= total_elements) break;

		scratch_bak[1]++;
		scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
		offset = scratch_bak[0] % fcols;
		coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
	}

	scratch_bak[2] = 0;
	scratch_bak[4] = 3;
	scratch_bak[5] = CUR_INFO.scratch[5] ^ 0x01;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	if(CUR_INFO.scratch[5]) {
		PRINTF("\r\nBuffering into proper location");
		for(uint i = CUR_INFO.scratch[6]; i < rows; i = ++CUR_INFO.scratch[6]) {
			for(uint j = CUR_INFO.scratch[7]; j < cols; j = ++CUR_INFO.scratch[7]) {
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_INFO.scratch[7] = 0;
		}
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);	
}

void task_sm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(total_elements % 2 == 0 && pos % 2 == 0) { // A
		dest = inter;
		inter = tmp;
	} else if(total_elements % 2 == 1 && pos % 2 == 1) { // B
		dest = inter;
		inter = tmp;
	}

	uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	uint n = idx % fcols; // Cols
	for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
		for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
			fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
			if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
				w = 0;
			}
			if(zero == 2) {
				w = F_ADD(w, MAT_GET(inter, i, j));
			}
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[3] = 0;
	}
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void __attribute__((interrupt(DMA_VECTOR)))dmaIsrHandler(void) {
    switch (__even_in_range(DMAIV, DMAIV_DMA2IFG)) {
    	case DMAIV_DMA0IFG:
        case DMAIV_DMA1IFG:
        case DMAIV_DMA2IFG:
            break;
        default: 
          break;
   }
   __bic_SR_register_on_exit(LPM0_bits);
}

void __attribute__ ((interrupt(LEA_VECTOR))) msp_lea_isr(void) {
    /* Save the interrupt flags, clear interrupt and exit LPM0. */
    uint16_t flags = LEAIFG;
    LEAIFG |= flags;
    msp_lea_ifg = flags;
    __bic_SR_register_on_exit(LPM0_bits);
}