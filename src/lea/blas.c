#include <string.h>
#include <libio/console.h>
#include <libmspdriver/driverlib.h>
#include <libdsp/DSPLib.h>
#include <libalpaca/alpaca.h>

#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "buffer.h"
#include "fixed.h"
#include "mat.h"
#include "misc.h"
#include "profile.h"

static __fram mat_t m1 = {.data = MAT_BUFFER(0)};
static __fram mat_t m2 = {.data = MAT_BUFFER(1)};
static __fram mat_t *inter1 = &m1;
static __fram mat_t *inter2 = &m2;

static DSPLIB_DATA(tsrc1, 2) fixed tsrc1[CONFIG_TILE_SIZE];
static DSPLIB_DATA(tsrc2, 2) fixed tsrc2[CONFIG_TILE_SIZE];
static DSPLIB_DATA(tsrc3, 2) fixed tsrc3[CONFIG_MAT_BUF_SIZE];
static DSPLIB_DATA(tsrc4, 2) fixed tsrc4[CONFIG_MAT_BUF_SIZE];
static DSPLIB_DATA(tdest, 2) fixed tdest[2];

// static __fram fixed tsrc1[CONFIG_TILE_SIZE];
// static __fram fixed tsrc2[CONFIG_TILE_SIZE];
// static __fram fixed tsrc3[CONFIG_MAT_BUF_SIZE];
// static __fram fixed tsrc4[CONFIG_MAT_BUF_SIZE];
// static __fram fixed tdest[2];



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

uint greatest_tile_size(uint dim, uint max);
void DMA_startSleepTransfer(uint channel);
void check_calibrate();
void task_calibrate();
TASK(TASK_UID_BLAS_OFFSET + 12, task_calibrate);

// Resets a task
static __fram task_t *last_task;
void task_cleanup_blas() {
	// PRINTF("\r\n     Finishing BLAS");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
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

void check_calibrate(){
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
	TASK_REF(task_calibrate)->info.return_task = CUR_TASK;
	TRANSITION_TO(task_calibrate);
}

void task_debug2();
void pulse(uint pin) {
	P6DIR = 0x03;
	P6OUT = pin;
	__delay_cycles(0x100);
	P6OUT = 0x00;
}

__known fixed debug_area2[10] = {0xadde};
TASK(50, task_debug2);
void task_debug2() {
	P1OUT = 0x00;
#ifdef CONFIG_INTERMITTENT
	while(1){}
#else
	PRINTF("\r\n debugging...");
	exit(0);
#endif
}

void task_calibrate() {
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
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);
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
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_mul(1);
			inc_ld(2);
			inc_st(1);
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
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_st(1);
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
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);
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
	check_calibrate();
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
		if(common_tile_size > 12 DMA_ENABLE) {
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
		    inc_addr_add(1);
		    inc_addr_mul(1);
		    inc_ld_vec(common_tile_size);
		} else {
			inc_addr_add(1);
		    inc_addr_mul(1);
		    inc_ld(common_tile_size);
			memcpy(tsrc1, MAT_PTR(filter, i, j), sizeof(fixed) * common_tile_size);
		}
		for(uint k = CUR_INFO.scratch[4]; k < dcols; k = ++CUR_INFO.scratch[4]) {
			inc_addr_mul(2);
			inc_addr_add(2);
			inc_mac_vec(common_tile_size);
			inc_st(1);
			if(zero == 2) {
				inc_add(1);
				inc_addr_add(1);
				inc_addr_mul(1);
				inc_ld(1);
			}
			if(common_tile_size > 12 DMA_ENABLE) {
				// DMA Src
				inc_ld_vec(common_tile_size);
				DMA_setTransferSize(dmaConfig[1].channelSelect, common_tile_size);
			    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
								(uint32_t) MAT_PTR(src, k, j),
	                      		DMA_DIRECTION_INCREMENT);
			    DMA_setDstAddress(dmaConfig[1].channelSelect,
			                    (uint32_t) (tsrc2),
			                    DMA_DIRECTION_INCREMENT);
				DMA_enableTransfers(dmaConfig[1].channelSelect);
			    DMA_startSleepTransfer(dmaConfig[1].channelSelect);
			} else {
				inc_ld(common_tile_size);
				memcpy(tsrc2, MAT_PTR(src, k, j), sizeof(fixed) * common_tile_size);
			}
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
	inc_addr_add(1);
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
__fram fixed coalesced_filter[CONFIG_MAT_BUF_SIZE];
void task_dm_conv() {
	check_calibrate();
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint srows = MAT_GET_DIM(src, 1);
	uint scols = MAT_GET_DIM(src, 2);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	uint common_tile_size = greatest_tile_size(srows * scols, tile_size);
	uint common_tile_size_rows = common_tile_size / cols;
	if(common_tile_size_rows == 0) common_tile_size_rows = 1;
	common_tile_size = common_tile_size_rows * MAT_GET_DIM(src, 2);
	MAT_RESHAPE(inter1, rows, cols);

	uint k = CUR_INFO.scratch[0];
	uint l = CUR_INFO.scratch[1];
	uint remaining_rows = (common_tile_size_rows < frows - l) ? common_tile_size_rows : frows - l;
	common_tile_size = remaining_rows * scols;
	// PRINTF("\r\n k: %u l: %u tile_size: %u remaining: %u frows: %u flayers: %u", 
		// k, l, tile_size, remaining_rows, frows, flayers);

	char state = CUR_INFO.scratch[4];
	if(state == 0) {
		scratch_bak[4] = 1;
		scratch_bak[5] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
		transition_to(CUR_TASK);
	} else if(state == 1 || state == 2) { // generate filter
		memset(coalesced_filter, 0, sizeof(fixed) * (scols + common_tile_size));
		coalesced_filter[(cols + common_tile_size)] = 0;
		for(uint i = CUR_INFO.scratch[2]; i < remaining_rows; i = ++CUR_INFO.scratch[2]) {
			memcpy(coalesced_filter + scols * (i + 1), MAT_PTR(filter, k, l + i, 0), sizeof(fixed) * fcols);
		}
		scratch_bak[2] = 0;
		scratch_bak[4] = (state == 1) ? 3 : 4;
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(CUR_INFO.scratch[5]) { // Swap
		dest = inter;
		inter = tmp;
	}

	if(scols + common_tile_size > 12) {
		DMA_setTransferSize(dmaConfig[1].channelSelect, scols + common_tile_size);
	    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
						(uint32_t) coalesced_filter,
                  		DMA_DIRECTION_INCREMENT);
	    DMA_setDstAddress(dmaConfig[1].channelSelect,
	                    (uint32_t) tsrc1,
	                    DMA_DIRECTION_INCREMENT);
		DMA_enableTransfers(dmaConfig[1].channelSelect);
		DMA_startSleepTransfer(dmaConfig[1].channelSelect);

		DMA_setTransferSize(dmaConfig[1].channelSelect, scols + common_tile_size);
	    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
						(uint32_t) coalesced_filter,
                  		DMA_DIRECTION_INCREMENT);
	    DMA_setDstAddress(dmaConfig[1].channelSelect,
	                    (uint32_t) (tsrc2 + 1),
	                    DMA_DIRECTION_INCREMENT);
		DMA_enableTransfers(dmaConfig[1].channelSelect);
		DMA_startSleepTransfer(dmaConfig[1].channelSelect);
	} else {
		memcpy(tsrc1, coalesced_filter, sizeof(fixed) * (scols + common_tile_size));
		memcpy(tsrc2 + 1, coalesced_filter, sizeof(fixed) * (scols + common_tile_size));
	}

    msp_mac_q15_params params = {.length = common_tile_size + common_tile_size % 2};

	msp_status status;
    uint dma_i = CUR_INFO.scratch[2];
    if(common_tile_size > 12) {
    	DMA_setTransferSize(dmaConfig[1].channelSelect, common_tile_size);
	    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
						(uint32_t) MAT_PTR(src, k, dma_i + l, 0),
                  		DMA_DIRECTION_INCREMENT);
	    DMA_setDstAddress(dmaConfig[1].channelSelect,
	                    (uint32_t) tsrc3,
	                    DMA_DIRECTION_INCREMENT);
		DMA_enableTransfers(dmaConfig[1].channelSelect);
		DMA_startSleepTransfer(dmaConfig[1].channelSelect);
    } else {
    	memcpy(tsrc3, MAT_PTR(src, k, dma_i + l, 0), sizeof(fixed) * common_tile_size);
    }
    dma_i += remaining_rows;
	for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
		if(i + l + remaining_rows >= dma_i) {
			if(scols > 12) {
				DMA_setTransferSize(dmaConfig[1].channelSelect, scols);
			    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
								(uint32_t) MAT_PTR(src, k, dma_i + l, 0),
		                  		DMA_DIRECTION_INCREMENT);
			    DMA_setDstAddress(dmaConfig[1].channelSelect,
			                    (uint32_t) (tsrc3 + dma_i * scols),
			                    DMA_DIRECTION_INCREMENT);
				DMA_enableTransfers(dmaConfig[1].channelSelect);
				DMA_startSleepTransfer(dmaConfig[1].channelSelect);
			} else {
				memcpy(tsrc3 + dma_i * scols, MAT_PTR(src, k, dma_i + l, 0), sizeof(fixed) * scols);
			}
		    dma_i++;
		}
		for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
		    fixed w = 0;
		    if(state == 4) w = MAT_GET(inter, i, j);
		    if(j & 0x1) {
		    	status = msp_mac_q15(&params, tsrc2 - j + 1 + scols, tsrc3 + i * scols, tdest);
		    } else {
				status = msp_mac_q15(&params, tsrc1 - j + scols, tsrc3 + i * scols, tdest);
	    	}
			fixed rounded = ((*tdest >> 1) + F_K) >> F_N;
    		w = F_ADD(w, rounded);
    		MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[3] = 0;
	}
	// coalesce a filter
	scratch_bak[0] = (l + remaining_rows == frows)? k + 1 : k; // flayers
	scratch_bak[1] = (l + remaining_rows == frows)? 0 : l + remaining_rows; // frows

	scratch_bak[2] = 0; // i
	scratch_bak[4] = 2; // state machine
	scratch_bak[5] = CUR_INFO.scratch[5] ^ 0x01; // swap buffer? 
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(!(k + 1 == flayers && l + remaining_rows == frows)) transition_to(CUR_TASK);
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

void task_dm_conv_same() {
	check_calibrate();
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint srows = MAT_GET_DIM(src, 1);
	uint scols = MAT_GET_DIM(src, 2);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	uint common_tile_size = greatest_tile_size(srows * scols, tile_size);
	uint common_tile_size_rows = common_tile_size / cols;
	if(common_tile_size_rows == 0) common_tile_size_rows = 1;
	common_tile_size = common_tile_size_rows * MAT_GET_DIM(src, 2);
	MAT_RESHAPE(inter1, rows, cols);

	uint k = CUR_INFO.scratch[0];
	uint l = CUR_INFO.scratch[1];
	uint remaining_rows = (common_tile_size_rows < frows - l) ? common_tile_size_rows : frows - l;
	common_tile_size = remaining_rows * scols;
	// PRINTF("\r\n k: %u l: %u tile_size: %u remaining: %u frows: %u flayers: %u", 
		// k, l, tile_size, remaining_rows, frows, flayers);

	char state = CUR_INFO.scratch[4];
	if(state == 0) {
		scratch_bak[4] = 1;
		scratch_bak[5] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
		transition_to(CUR_TASK);
	} else if(state == 1 || state == 2) { // generate filter
		memset(coalesced_filter, 0, sizeof(fixed) * (scols + common_tile_size));
		coalesced_filter[(cols + common_tile_size)] = 0;
		for(uint i = CUR_INFO.scratch[2]; i < remaining_rows; i = ++CUR_INFO.scratch[2]) {
			memcpy(coalesced_filter + scols * (i + 1), MAT_PTR(filter, k, l + i, 0), sizeof(fixed) * fcols);
		}
		scratch_bak[2] = 0;
		scratch_bak[4] = (state == 1) ? 3 : 4;
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(CUR_INFO.scratch[5]) { // Swap
		dest = inter;
		inter = tmp;
	}

	if(scols + common_tile_size > 12) {
		DMA_setTransferSize(dmaConfig[1].channelSelect, scols + common_tile_size);
	    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
						(uint32_t) coalesced_filter,
                  		DMA_DIRECTION_INCREMENT);
	    DMA_setDstAddress(dmaConfig[1].channelSelect,
	                    (uint32_t) tsrc1,
	                    DMA_DIRECTION_INCREMENT);
		DMA_enableTransfers(dmaConfig[1].channelSelect);
		DMA_startSleepTransfer(dmaConfig[1].channelSelect);

		DMA_setTransferSize(dmaConfig[1].channelSelect, scols + common_tile_size);
	    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
						(uint32_t) coalesced_filter,
                  		DMA_DIRECTION_INCREMENT);
	    DMA_setDstAddress(dmaConfig[1].channelSelect,
	                    (uint32_t) (tsrc2 + 1),
	                    DMA_DIRECTION_INCREMENT);
		DMA_enableTransfers(dmaConfig[1].channelSelect);
		DMA_startSleepTransfer(dmaConfig[1].channelSelect);
	} else {
		memcpy(tsrc1, coalesced_filter, sizeof(fixed) * (scols + common_tile_size));
		memcpy(tsrc2 + 1, coalesced_filter, sizeof(fixed) * (scols + common_tile_size));
	}

    msp_mac_q15_params params = {.length = common_tile_size + common_tile_size % 2};

	msp_status status;
    uint dma_i = CUR_INFO.scratch[2];
    if(common_tile_size > 12) {
    	DMA_setTransferSize(dmaConfig[1].channelSelect, common_tile_size);
	    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
						(uint32_t) MAT_PTR(src, k, dma_i + l, 0),
                  		DMA_DIRECTION_INCREMENT);
	    DMA_setDstAddress(dmaConfig[1].channelSelect,
	                    (uint32_t) tsrc3,
	                    DMA_DIRECTION_INCREMENT);
		DMA_enableTransfers(dmaConfig[1].channelSelect);
		DMA_startSleepTransfer(dmaConfig[1].channelSelect);
    } else {
    	memcpy(tsrc3, MAT_PTR(src, k, dma_i + l, 0), sizeof(fixed) * common_tile_size);
    }
    dma_i += remaining_rows;
	for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
		if(i + l + remaining_rows >= dma_i) {
			if(scols > 12) {
				DMA_setTransferSize(dmaConfig[1].channelSelect, scols);
			    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
								(uint32_t) MAT_PTR(src, k, dma_i + l, 0),
		                  		DMA_DIRECTION_INCREMENT);
			    DMA_setDstAddress(dmaConfig[1].channelSelect,
			                    (uint32_t) (tsrc3 + dma_i * scols),
			                    DMA_DIRECTION_INCREMENT);
				DMA_enableTransfers(dmaConfig[1].channelSelect);
				DMA_startSleepTransfer(dmaConfig[1].channelSelect);
			} else {
				memcpy(tsrc3 + dma_i * scols, MAT_PTR(src, k, dma_i + l, 0), sizeof(fixed) * scols);
			}
		    dma_i++;
		}
		for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
		    fixed w = 0;
		    if(state == 4) w = MAT_GET(inter, i, j);
		    // Make part of the filter zero for same padding
		    for(uint k = scols; k < common_tile_size + common_tile_size % 2; k += scols) {
			    *(tsrc2 - j + 1 + k + fcols) = 0;
		    }
		    if(j & 0x1) {
		    	status = msp_mac_q15(&params, tsrc2 - j + 1 + scols, tsrc3 + i * scols, tdest);
		    } else {
				status = msp_mac_q15(&params, tsrc1 - j + scols, tsrc3 + i * scols, tdest);
	    	}
			fixed rounded = ((*tdest >> 1) + F_K) >> F_N;
    		w = F_ADD(w, rounded);
    		MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[3] = 0;
	}
	// coalesce a filter
	scratch_bak[0] = (l + remaining_rows == frows)? k + 1 : k; // flayers
	scratch_bak[1] = (l + remaining_rows == frows)? 0 : l + remaining_rows; // frows

	scratch_bak[2] = 0; // i
	scratch_bak[4] = 2; // state machine
	scratch_bak[5] = CUR_INFO.scratch[5] ^ 0x01; // swap buffer? 
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(!(k + 1 == flayers && l + remaining_rows == frows)) transition_to(CUR_TASK);
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
		inc_addr_add(3);
		inc_addr_mul(2);
		inc_mul(1);
		inc_ld(2);
		inc_st(1);
		if(zero == 2) {
			inc_add(1);
			inc_addr_mul(1);
			inc_addr_add(1);
			inc_ld(1);
			w = F_ADD(w, MAT_GET(inter, i, j));
		}
		MAT_SET(dest, w, i, j);
		write_to_gbuf((uint8_t *)(dest->data + i * dcols + j), (uint8_t *)(inter->data + i * dcols + j), sizeof(uint));
	}

	scratch_bak[0] = pos + 1;
	scratch_bak[2] = k + filter->sparse.offsets[pos + 1];
	scratch_bak[3] = (scratch_bak[2] / cols > 0) ? 1 : 2;
	scratch_bak[1] = i + scratch_bak[2] / cols;
	inc_addr_add(3);
	inc_addr_mul(1);
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

#if 1
void task_sm_conv() {
	check_calibrate();
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
		inc_addr_mul(1);
		inc_addr_add(3);
		coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
		for(uint i = 0; i < fcols; i++) {
			if((scratch_bak[0] + filter->sparse.offsets[scratch_bak[1] + 1]) / fcols != init_row || 
				scratch_bak[1] + 1 >= total_elements) break;

			scratch_bak[1]++;
			scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
			inc_addr_add(5);
			inc_ld(1);
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
	inc_addr_mul(2);
	// PRINTF("\r\n tile_size: %u length: %u tapLength: %u k: %u l: %u", 
		// common_tile_size, common_tile_size + even_offset, fcols + fcols % 2, k, l);

	msp_status status;
	uint length = common_tile_size + even_offset;
	if(cols < length) {
		length = cols + cols % 2;
	}
	msp_fir_q15_params params_fir = {.length = length, 
									.tapLength = (fcols + fcols % 2), .coeffs = tsrc1};
	msp_add_q15_params params_add = {.length = length};
	uint fsize = fcols + fcols % 2;
	if(fsize > 12 DMA_ENABLE) {
		inc_ld_vec(fsize);
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
		inc_ld(fcols + fcols % 2);
		memcpy(tsrc1, coalesced_filter, sizeof(fixed) * (fcols + fcols % 2));
	}

	uint copy_length = common_tile_size;
	if(common_tile_size < scols) {
		if(fcols > copy_length) {
			copy_length = fcols;
		} else {
			copy_length += fcols;
		}
	}
	if(common_tile_size > 12 DMA_ENABLE) {
		for(uint i = CUR_INFO.scratch[2]; i < stride[1] * rows; i = (CUR_INFO.scratch[2] += stride[1])) {
			uint idx_i = (stride[1] > 1) ? i / stride[1] : i;
			for(uint j = CUR_INFO.scratch[3]; j < scols; j = (CUR_INFO.scratch[3] += common_tile_size)) {
				inc_ld_vec(copy_length);
				inc_addr_mul(1);
				inc_addr_add(2);
				DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
			    DMA_setSrcAddress(dmaConfig[0].channelSelect,
			                      (uint32_t) (MAT_PTR(src, k, i + l, j)),
			                      DMA_DIRECTION_INCREMENT);

			    DMA_setDstAddress(dmaConfig[0].channelSelect,
			                      (uint32_t) (tsrc3),
			                      DMA_DIRECTION_INCREMENT);
			    DMA_enableTransfers(dmaConfig[0].channelSelect);
			    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
				inc_fir_vec(length, fcols + fcols % 2);
				if(even_offset) tsrc3[copy_length + even_offset] = 0;
				status = msp_fir_q15(&params_fir, tsrc3, tsrc2);
				if(state == 3) {
					inc_addr_add(1);
					inc_addr_mul(1);
					inc_ld_vec(copy_length);
					inc_add_vec(length);
					DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(inter, idx_i, j)),
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (tsrc4),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
					if(even_offset) tsrc4[copy_length + even_offset] = 0;
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st_vec(copy_length);
					status = msp_add_q15(&params_add, tsrc2, tsrc4, tsrc3);
					DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) tsrc3,
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(dest, idx_i, j)),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
				} else {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st_vec(copy_length);
					DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) tsrc2,
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(dest, idx_i, j)),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
				}
			}
			CUR_INFO.scratch[3] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[2]; i < rows * stride[1]; i = (CUR_INFO.scratch[2] += stride[1])) {
			uint idx_i = (stride[1] > 1) ? i / stride[1] : i;
			for(uint j = CUR_INFO.scratch[3]; j < scols; j = (CUR_INFO.scratch[3] += common_tile_size)) {
				inc_addr_mul(1);
				inc_addr_add(1);
				inc_ld(copy_length);
				memcpy(tsrc3, MAT_PTR(src, k, i + l, j), sizeof(fixed) * copy_length);
				if(even_offset) tsrc3[copy_length + even_offset] = 0;
				inc_fir_vec(length, fcols + fcols % 2);
				status = msp_fir_q15(&params_fir, tsrc3, tsrc2);
				if(state == 3) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_ld(copy_length);
					memcpy(tsrc4, MAT_PTR(inter, idx_i, j), sizeof(fixed) * copy_length);
					if(even_offset) tsrc4[copy_length + even_offset] = 0;
					inc_add_vec(length);
					status = msp_add_q15(&params_add, tsrc2, tsrc4, tsrc3);
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st(copy_length);
					memcpy(MAT_PTR(dest, idx_i, j), tsrc3, sizeof(fixed) * copy_length);
				} else {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st(copy_length);
					memcpy(MAT_PTR(dest, idx_i, j), tsrc2, sizeof(fixed) * copy_length);
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
	inc_addr_mul(1);
	inc_addr_add(5);
	coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
	for(uint i = 0; i < fcols; i++) {
		if((scratch_bak[0] + filter->sparse.offsets[scratch_bak[1] + 1]) / fcols != init_row || 
			scratch_bak[1] + 1 >= total_elements) break;

		inc_addr_add(5);
		inc_ld(1);
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
				inc_addr_add(2);
				inc_addr_mul(2);
				inc_st(1);
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
	check_calibrate();
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
		inc_addr_mul(1);
		inc_addr_add(3);
		coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
		for(uint i = 0; i < fcols; i++) {
			if((scratch_bak[0] + filter->sparse.offsets[scratch_bak[1] + 1]) / fcols != init_row || 
				scratch_bak[1] + 1 >= total_elements) break;

			scratch_bak[1]++;
			scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
			inc_addr_add(5);
			inc_ld(1);
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
	inc_addr_mul(2);
	// PRINTF("\r\n tile_size: %u length: %u tapLength: %u k: %u l: %u", 
		// common_tile_size, common_tile_size + even_offset, fcols + fcols % 2, k, l);

	uint length = common_tile_size + even_offset;
	if(cols < length) {
		length = cols + cols % 2;
	}

	msp_status status;
	msp_fir_q15_params params_fir = {.length = length + even_offset, 
									.tapLength = (fcols + fcols % 2), .coeffs = tsrc1};
	msp_add_q15_params params_add = {.length = length};
	uint fsize = fcols + fcols % 2;
	if(fsize > 12 DMA_ENABLE) {
		inc_ld_vec(fsize);
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
		inc_ld(fcols + fcols % 2);
		memcpy(tsrc1, coalesced_filter, sizeof(fixed) * fsize);
	}

	uint copy_length = common_tile_size;
	if(common_tile_size < scols) {
		if(fcols > copy_length) {
			copy_length = fcols;
		} else {
			copy_length += fcols;
		}
	}

	if(common_tile_size > 12 DMA_ENABLE) {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			if(i + l > rows) {
				memset(tsrc1, 0, sizeof(fixed) * fsize);
			}
			for(uint j = CUR_INFO.scratch[3]; j < scols; j = (CUR_INFO.scratch[3] += common_tile_size)) {
				inc_ld_vec(common_tile_size);
				inc_addr_mul(1);
				inc_addr_add(2);
				DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
			    DMA_setSrcAddress(dmaConfig[0].channelSelect,
			                      (uint32_t) (MAT_PTR(src, k, i + l, j)),
			                      DMA_DIRECTION_INCREMENT);

			    DMA_setDstAddress(dmaConfig[0].channelSelect,
			                      (uint32_t) (tsrc3),
			                      DMA_DIRECTION_INCREMENT);
			    DMA_enableTransfers(dmaConfig[0].channelSelect);
			    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
			    inc_fir_vec(length, fcols + fcols % 2);
				if(even_offset) tsrc3[copy_length + even_offset] = 0;
				status = msp_fir_q15(&params_fir, tsrc3, tsrc2);
				if(state == 3) {
					inc_addr_add(1);
					inc_addr_mul(1);
					inc_ld_vec(copy_length);
					inc_add_vec(copy_length);
					DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(inter, i, j)),
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (tsrc4),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
					if(even_offset) tsrc4[copy_length + even_offset] = 0;
					inc_add_vec(length);
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st_vec(copy_length);
					status = msp_add_q15(&params_add, tsrc2, tsrc4, tsrc3);
					DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
				    DMA_setSrcAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) tsrc3,
				                      DMA_DIRECTION_INCREMENT);

				    DMA_setDstAddress(dmaConfig[0].channelSelect,
				                      (uint32_t) (MAT_PTR(dest, i, j)),
				                      DMA_DIRECTION_INCREMENT);
				    DMA_enableTransfers(dmaConfig[0].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[0].channelSelect);
				} else {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st_vec(copy_length);
					DMA_setTransferSize(dmaConfig[0].channelSelect, copy_length);
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
				inc_addr_mul(1);
				inc_addr_add(1);
				inc_ld(common_tile_size);
				memcpy(tsrc3, MAT_PTR(src, k, i + l, j), sizeof(fixed) * copy_length);
				if(even_offset) tsrc3[common_tile_size + even_offset] = 0;
				inc_fir_vec(length, fcols + fcols % 2);
				status = msp_fir_q15(&params_fir, tsrc3, tsrc2);
				if(state == 3) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_ld(copy_length);
					memcpy(tsrc4, MAT_PTR(inter, i, j), sizeof(fixed) * copy_length);
					if(even_offset) tsrc4[copy_length + even_offset] = 0;
					inc_add_vec(length);
					status = msp_add_q15(&params_add, tsrc2, tsrc4, tsrc3);
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st(copy_length);
					memcpy(MAT_PTR(dest, i, j), tsrc3, sizeof(fixed) * copy_length);
				} else {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_st(copy_length);
					memcpy(MAT_PTR(dest, i, j), tsrc2, sizeof(fixed) * copy_length);
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
	inc_addr_mul(1);
	inc_addr_add(5);
	coalesced_filter[fcols - offset - feven_offset] = MAT_GET(filter, scratch_bak[1]) << SHIFT;
	for(uint i = 0; i < fcols; i++) {
		if((scratch_bak[0] + filter->sparse.offsets[scratch_bak[1] + 1]) / fcols != init_row || 
			scratch_bak[1] + 1 >= total_elements) break;

		inc_addr_add(5);
		inc_ld(1);
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
				inc_addr_add(2);
				inc_addr_mul(2);
				inc_st(1);
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_INFO.scratch[7] = 0;
		}
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

#else

void task_sm_conv() {
	check_calibrate();
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	uint common_tile_size = greatest_tile_size(fcols, tile_size);
	uint common_tile_size_cols = greatest_tile_size(MAT_GET_DIM(src, 1), tile_size);
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		memset(coalesced_filter, 0, sizeof(filter) * common_tile_size);
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[1] = pos;
		uint offset = scratch_bak[0] % common_tile_size;
		inc_addr_add(1);
		inc_ld(1);
		coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
		for(uint i = 0; i < common_tile_size; i++) {
			if(offset + filter->sparse.offsets[scratch_bak[1] + 1] < common_tile_size
				&& scratch_bak[1] + 1 < total_elements) {
				scratch_bak[1]++;
				scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
				offset = scratch_bak[0] % common_tile_size;
				inc_addr_add(3);
				inc_ld(1);
				coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
			}
		}
		coalesced_filter[common_tile_size] = 0;
		scratch_bak[4] = 1;
		scratch_bak[5] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(CUR_INFO.scratch[5]) { // Swap
		dest = inter;
		inter = tmp;
	}

	inc_ld_vec(common_tile_size + common_tile_size % 2);
	DMA_setTransferSize(dmaConfig[0].channelSelect, common_tile_size + common_tile_size % 2);
    DMA_setSrcAddress(dmaConfig[0].channelSelect,
                      (uint32_t) (coalesced_filter),
                      DMA_DIRECTION_INCREMENT);

    DMA_setDstAddress(dmaConfig[0].channelSelect,
                      (uint32_t) (tsrc1),
                      DMA_DIRECTION_INCREMENT);
    DMA_enableTransfers(dmaConfig[0].channelSelect);
    DMA_startSleepTransfer(dmaConfig[0].channelSelect);

    msp_mac_q15_params params = {.length = common_tile_size + common_tile_size % 2};
    // Use algined idx here
    uint aligned_idx = idx - idx % common_tile_size;
    uint k = aligned_idx / (fcols * frows); // Layers
	uint l = (aligned_idx % (fcols * frows)) / fcols; // Rows
	uint n = aligned_idx % fcols; // Cols
	inc_addr_mul(2);
	// PRINTF("\r\nFilter: ");
	// for(uint i = 0; i < common_tile_size + 1; i++) {
		// PRINTF("%i ", tsrc1[i]);
	// }
	uint load_size = (common_tile_size > common_tile_size_cols) ? common_tile_size : common_tile_size_cols;
	for(uint i = CUR_INFO.scratch[2]; i < rows * stride[1]; i = (CUR_INFO.scratch[2] += stride[1])) {
		uint dma_j = CUR_INFO.scratch[3] + n;
		uint dma_offset = 0;
		uint dma_pos = 0;
		uint idx_i = (stride[1] > 1) ? i / stride[1] : i;
		for(uint j = CUR_INFO.scratch[3]; j < cols * stride[2]; j = (CUR_INFO.scratch[3] += stride[2])) {
			uint idx_j = (stride[2] > 1) ? j / stride[2] : j;
			if(j + n + common_tile_size >= dma_j) {
				if(load_size <= 12) {
					inc_addr_add(2);
					inc_addr_mul(2);
					inc_ld(load_size);
					inc_ld(load_size);
					memcpy(tsrc2 + dma_offset, MAT_PTR(src, k, i + l, dma_j), sizeof(fixed) * load_size);
					memcpy(tsrc3 + dma_offset + 1, MAT_PTR(src, k, i + l, dma_j), sizeof(fixed) * load_size);
				} else {
					inc_addr_add(2);
					inc_addr_mul(2);
					inc_ld_vec(load_size);
					inc_ld_vec(load_size);
				    DMA_setTransferSize(dmaConfig[1].channelSelect, load_size);
				    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
									(uint32_t) MAT_PTR(src, k, i + l, dma_j),
		                      		DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dmaConfig[1].channelSelect,
				                    (uint32_t) (tsrc2 + dma_offset),
				                    DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dmaConfig[1].channelSelect);

				    DMA_setTransferSize(dmaConfig[2].channelSelect, load_size);
				    DMA_setSrcAddress(dmaConfig[2].channelSelect, 
									(uint32_t) MAT_PTR(src, k, i + l, dma_j),
		                      		DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dmaConfig[2].channelSelect,
				                    (uint32_t) (tsrc3 + dma_offset + 1),
				                    DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dmaConfig[2].channelSelect);

					DMA_startSleepTransfer(dmaConfig[1].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[2].channelSelect);
				}
				inc_addr_add(2);
			    dma_offset += load_size;
			    dma_j += load_size;
			}
			if(zero == 2) {
				inc_ld(1);
				inc_addr_add(1);
				inc_addr_mul(1);
			}
		    fixed w = 0;
		    if(zero == 2) w = MAT_GET(inter, idx_i, idx_j);
		    if(j & 0x1) {
		    	inc_mac_vec(common_tile_size);
		    	msp_mac_q15(&params, tsrc1, tsrc3 + dma_pos + 2, tdest);
	    		dma_pos += 2;
		    } else {
				inc_mac_vec(common_tile_size);
				msp_mac_q15(&params, tsrc1, tsrc2 + dma_pos, tdest);
			}
			inc_add(2);
			inc_st(1);
			inc_addr_add(1);
			inc_addr_mul(1);
			fixed rounded = ((*tdest >> 1) + F_K) >> F_N;
    		w = F_ADD(w, rounded);
    		MAT_SET(dest, w, idx_i, idx_j);
		}
		CUR_INFO.scratch[3] = 0;
	}
	// coalesce a filter
	memset(coalesced_filter, 0, sizeof(filter) * common_tile_size);
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	inc_addr_add(3);
	inc_addr_mul(1);
	inc_ld(1);
	uint offset = scratch_bak[0] % common_tile_size;
	coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
	for(uint i = 0; i < common_tile_size; i++) {
		if(offset + filter->sparse.offsets[scratch_bak[1] + 1] < common_tile_size 
			&& scratch_bak[1] + 1 < total_elements) {
			scratch_bak[1]++;
			scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
			offset = scratch_bak[0] % common_tile_size;
			inc_addr_add(3);
			inc_ld(1);
			coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
		}
	}
	coalesced_filter[common_tile_size] = 0;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
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
				inc_ld(1);
				inc_st(1);
				inc_addr_mul(2);
				inc_addr_add(2);
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
	check_calibrate();
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	uint common_tile_size = greatest_tile_size(fcols, tile_size);
	uint common_tile_size_cols = greatest_tile_size(MAT_GET_DIM(src, 1), tile_size);
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		memset(coalesced_filter, 0, sizeof(filter) * common_tile_size);
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[1] = pos;
		uint offset = scratch_bak[0] % common_tile_size;
		inc_addr_add(1);
		inc_ld(1);
		coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
		for(uint i = 0; i < common_tile_size; i++) {
			if(offset + filter->sparse.offsets[scratch_bak[1] + 1] < common_tile_size
				&& scratch_bak[1] + 1 < total_elements) {
				scratch_bak[1]++;
				scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
				offset = scratch_bak[0] % common_tile_size;
				inc_addr_add(3);
				inc_ld(1);
				coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
			}
		}
		coalesced_filter[common_tile_size] = 0;
		scratch_bak[4] = 1;
		scratch_bak[5] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(CUR_INFO.scratch[5]) { // Swap
		dest = inter;
		inter = tmp;
	}

	inc_ld_vec(common_tile_size + common_tile_size % 2);
	DMA_setTransferSize(dmaConfig[0].channelSelect, common_tile_size + common_tile_size % 2);
    DMA_setSrcAddress(dmaConfig[0].channelSelect,
                      (uint32_t) (coalesced_filter),
                      DMA_DIRECTION_INCREMENT);

    DMA_setDstAddress(dmaConfig[0].channelSelect,
                      (uint32_t) (tsrc1),
                      DMA_DIRECTION_INCREMENT);
    DMA_enableTransfers(dmaConfig[0].channelSelect);
    DMA_startSleepTransfer(dmaConfig[0].channelSelect);

    msp_mac_q15_params params = {.length = common_tile_size + common_tile_size % 2};
    // Use algined idx here
    uint aligned_idx = idx - idx % common_tile_size;
    uint k = aligned_idx / (fcols * frows); // Layers
	uint l = (aligned_idx % (fcols * frows)) / fcols; // Rows
	uint n = aligned_idx % fcols; // Cols
	inc_addr_mul(2);
	// PRINTF("\r\nFilter: ");
	// for(uint i = 0; i < common_tile_size + 1; i++) {
		// PRINTF("%i ", tsrc1[i]);
	// }
	uint load_size = (common_tile_size > common_tile_size_cols) ? common_tile_size : common_tile_size_cols;
	for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
		uint dma_j = CUR_INFO.scratch[3] + n;
		uint dma_offset = 0;
		uint dma_pos = 0;
		for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
			if(j + n + common_tile_size >= dma_j) {
				if(load_size <= 12) {
					inc_addr_add(2);
					inc_addr_mul(2);
					inc_ld(load_size);
					inc_ld(load_size);
					memcpy(tsrc3 + dma_offset, MAT_PTR(src, k, i + l, dma_j), sizeof(fixed) * load_size);
					memcpy(tsrc4 + dma_offset + 1, MAT_PTR(src, k, i + l, dma_j), sizeof(fixed) * load_size);
				} else {
					inc_addr_add(2);
					inc_addr_mul(2);
					inc_ld_vec(load_size);
					inc_ld_vec(load_size);
				    DMA_setTransferSize(dmaConfig[1].channelSelect, load_size);
				    DMA_setSrcAddress(dmaConfig[1].channelSelect, 
									(uint32_t) MAT_PTR(src, k, i + l, dma_j),
		                      		DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dmaConfig[1].channelSelect,
				                    (uint32_t) (tsrc3 + dma_offset),
				                    DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dmaConfig[1].channelSelect);

				    DMA_setTransferSize(dmaConfig[2].channelSelect, load_size);
				    DMA_setSrcAddress(dmaConfig[2].channelSelect, 
									(uint32_t) MAT_PTR(src, k, i + l, dma_j),
		                      		DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dmaConfig[2].channelSelect,
				                    (uint32_t) (tsrc4 + dma_offset + 1),
				                    DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dmaConfig[2].channelSelect);

					DMA_startSleepTransfer(dmaConfig[1].channelSelect);
				    DMA_startSleepTransfer(dmaConfig[2].channelSelect);
				}
				inc_addr_add(2);
			    dma_offset += load_size;
			    dma_j += load_size;
			}
			if(zero == 2) {
				inc_ld(1);
				inc_addr_add(1);
				inc_addr_mul(1);
			}
		    fixed w = 0;
		    if(zero == 2) w = MAT_GET(inter, i, j);
		    if(j & 0x1) {
			    inc_mac_vec(common_tile_size);
		    	msp_mac_q15(&params, tsrc1, tsrc4 + dma_pos + 2, tdest);
	    		dma_pos += 2;
		    } else {
			    inc_mac_vec(common_tile_size);
				msp_mac_q15(&params, tsrc1, tsrc3 + dma_pos, tdest);
			}
			inc_add(2);
			inc_st(1);
			inc_addr_add(1);
			inc_addr_mul(1);
			fixed rounded = ((*tdest >> 1) + F_K) >> F_N;
			if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
				rounded = 0;
			}
    		w = F_ADD(w, rounded);
    		MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[3] = 0;
	}
	// coalesce a filter
	memset(coalesced_filter, 0, sizeof(filter) * common_tile_size);
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	inc_addr_add(3);
	inc_addr_mul(1);
	inc_ld(1);
	uint offset = scratch_bak[0] % common_tile_size;
	coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
	for(uint i = 0; i < common_tile_size; i++) {
		if(offset + filter->sparse.offsets[scratch_bak[1] + 1] < common_tile_size 
			&& scratch_bak[1] + 1 < total_elements) {
			scratch_bak[1]++;
			scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
			offset = scratch_bak[0] % common_tile_size;
			inc_addr_add(3);
			inc_ld(1);
			coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
		}
	}
	coalesced_filter[common_tile_size] = 0;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
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
				inc_ld(1);
				inc_st(1);
				inc_addr_mul(2);
				inc_addr_add(2);
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_INFO.scratch[7] = 0;
		}
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}
#endif

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

#ifndef MSP_DISABLE_LEA
void __attribute__ ((interrupt(LEA_VECTOR))) msp_lea_isr(void) {
    /* Save the interrupt flags, clear interrupt and exit LPM0. */
    uint16_t flags = LEAIFG;
    LEAIFG |= flags;
    msp_lea_ifg = flags;
    __bic_SR_register_on_exit(LPM0_bits);
}
#endif