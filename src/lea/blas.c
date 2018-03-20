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

static __fram mat_t m1 = {.data = MAT_BUFFER(0)};
static __fram mat_t *inter1 = &m1;
static DSPLIB_DATA(tsrc1, 2) fixed tsrc1[CONFIG_TILE_SIZE];
static DSPLIB_DATA(tsrc2, 2) fixed tsrc2[MAT_BUFFER_SIZE];
static DSPLIB_DATA(tsrc3, 2) fixed tsrc3[MAT_BUFFER_SIZE];
static DSPLIB_DATA(tdest, 2) fixed tdest[CONFIG_TILE_SIZE];

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
		debug_area2[2] = 0x01;
		write_to_gbuf((uint8_t *)(CUR_INFO.scratch + 1), (uint8_t *)(&tile_size), sizeof(uint));
		transition_to(CUR_TASK);
	} else if(CUR_INFO.scratch[0] == 2 && tile_size == 0) {
		scratch_bak[0] = 3;
		scratch_bak[1] = CUR_INFO.scratch[1] / 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		transition_to(CUR_TASK);	
	}
	debug_area2[2] = 0x02;
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
	check_calibrate();
	#pragma GCC warning "not yet implemented"
}

void task_dm_conv_same() {
	check_calibrate();
	#pragma GCC warning "not yet implemented"
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

__fram fixed coalesced_filter[CONFIG_TILE_SIZE];
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
		coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
		for(uint i = 0; i < common_tile_size; i++) {
			if(offset + filter->sparse.offsets[scratch_bak[1] + 1] < common_tile_size
				&& scratch_bak[1] + 1 < total_elements) {
				scratch_bak[1]++;
				scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
				offset = scratch_bak[0] % common_tile_size;
				coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
			}
		}
		coalesced_filter[common_tile_size] = 0;
		scratch_bak[4] = 1;
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
		    pulse(0x02);
			if(j + n + common_tile_size >= dma_j) {
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
			    dma_offset += load_size;
			    dma_j += load_size;
			}
		    fixed w = 0;
		    if(zero == 2) w = MAT_GET(inter, i, j);
		    if(j & 0x1) {
			    while((LEACNF1 & LEABUSY) || msp_lea_locked) __no_operation(); // Spin waiting for LEA
		    	msp_mac_q15(&params, tsrc1, tsrc3 + dma_pos + 2, tdest);
	    		dma_pos += 2;
		    } else {
			    while((LEACNF1 & LEABUSY) || msp_lea_locked) __no_operation(); // Spin waiting for LEA
				msp_mac_q15(&params, tsrc1, tsrc2 + dma_pos, tdest);
			}
			debug_area2[i * 24 + j] = *tdest;
			fixed rounded = ((*tdest >> 1) + F_K) >> F_N;
    		w = F_ADD(w, rounded);
    		MAT_SET(dest, w, i, j);
		    pulse(0x02);
		}
		CUR_INFO.scratch[3] = 0;
	}
	// coalesce a filter
	memset(coalesced_filter, 0, sizeof(filter) * common_tile_size);
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	uint offset = scratch_bak[0] % common_tile_size;
	coalesced_filter[offset] = MAT_GET(filter, scratch_bak[1]);
	for(uint i = 0; i < common_tile_size; i++) {
		if(offset + filter->sparse.offsets[scratch_bak[1] + 1] < common_tile_size 
			&& scratch_bak[1] + 1 < total_elements) {
			scratch_bak[1]++;
			scratch_bak[0] += filter->sparse.offsets[scratch_bak[1]];
			offset = scratch_bak[0] % common_tile_size;
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
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_INFO.scratch[7] = 0;
		}
	}
	pulse(0x01);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv_same() {
	#pragma GCC warning "not yet implemented"
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