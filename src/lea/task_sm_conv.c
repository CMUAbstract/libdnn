#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>
#include <libmspdriver/driverlib.h>
#include <libdsp/DSPLib.h>

#include "lea.h"
#include "mem.h"
#include "blas.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"
#include "profile.h"
#include "cleanup.h"

TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv);
static __fram mat_t buf1 = {.data = MAT_BUFFER(0)};
static __fram mat_t buf2 = {.data = MAT_BUFFER(1)};
static __fram mat_t *buffer1 = &buf1;
static __fram mat_t *buffer2 = &buf2;
static __fram fixed coalesced_filter[CONFIG_TILE_SIZE];

// Dense matrix multiplication
void task_sm_conv() {
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t tile_size = 0;
	uint16_t fcols = filter->sparse.dims[2];
	if(/*params.stride[1] + params.stride[2] == 2 && fcols != 1*/ 1) {
		tile_size = check_calibrate();
	}
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter1 = buffer1;
	mat_t *inter2 = dest;
	if(params.stride[1] + params.stride[2] != 2 && fcols != 1) {
		inter2 = buffer2;
	}

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);


	// LEA/DMA don't work well for strided convolution
	if(/*params.stride[1] + params.stride[2] != 2 || fcols == 1*/ 0) {
		MAT_RESHAPE(inter1, rows, cols);
		mat_t *tmp = dest;
		if(CUR_SCRATCH[3]) { // Swap buffers
			dest = inter1;
			inter1 = tmp;
		}
		uint16_t pos = CUR_SCRATCH[0];
		uint16_t idx = CUR_SCRATCH[1];
		bool zero = false;
		if(pos == 0) {
			zero = true;
			idx += filter->sparse.offsets[pos];
		}
		uint16_t k = idx / (fcols * frows); // Layers
		uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
		uint16_t n = idx % fcols; // Cols

		uint16_t i_stride = CUR_SCRATCH[4] / params.stride[1];
		uint16_t j_stride = CUR_SCRATCH[5] / params.stride[2];
		fixed f = MAT_GET(filter, pos);
		fixed *inter1_ptr = MAT_PTR(inter1, i_stride, j_stride);
		fixed *dest_ptr = MAT_PTR(dest, i_stride, j_stride);
		for(uint16_t i = CUR_SCRATCH[4]; 
			i < rows * params.stride[1]; i = (CUR_SCRATCH[4] += params.stride[1])){
			fixed *src_ptr = MAT_PTR(src, k, i + l, CUR_SCRATCH[5] + n);
			for(uint16_t j = CUR_SCRATCH[5]; 
				j < cols * params.stride[2]; j = (CUR_SCRATCH[5] += params.stride[2])){
				fixed w = 0;
				if(!params.same_padding || (i + l < MAT_GET_DIM(src, 1) && 
					j + n < MAT_GET_DIM(src, 2))) {
					w = F_MUL(f, (*src_ptr >> SHIFT));
				}
				if(!zero) {
					w = F_ADD(w, *inter1_ptr); // Zero
					inter1_ptr++;
				}
				*dest_ptr = w;
				dest_ptr++;
				src_ptr += params.stride[2];
			}
			CUR_SCRATCH[5] = 0;
		}

		scratch_bak[0] = pos + 1;
		scratch_bak[1] = idx + filter->sparse.offsets[pos + 1];

		scratch_bak[3] = CUR_SCRATCH[3] ^ 0x01;
		scratch_bak[4] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), 
			(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
		if(pos < total_elements - 1) {
			write_to_gbuf((uint8_t *)(scratch_bak + 3), 
				(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
			transition_to(CUR_TASK);
		}
		if(CUR_SCRATCH[3]) {
			for(uint16_t i = CUR_SCRATCH[6]; i < rows; i = (++CUR_SCRATCH[6])){
				for(uint16_t j = CUR_SCRATCH[7]; j < cols; j = (++CUR_SCRATCH[7])){
					MAT_SET(inter1, MAT_GET(dest, i, j), i, j);
				}
				CUR_SCRATCH[7] = 0;
			}
		}
		POP_STACK(mat_stack, 3);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	if(!params.same_padding && inter2 == dest) {
		rows = MAT_GET_DIM(src, 1);
		cols = MAT_GET_DIM(src, 2);
	}
	rows *= params.stride[1];
	cols *= params.stride[2];

	// PRINTF("\r\n rows: %u cols: %u", rows, cols);
	MAT_RESHAPE(inter1, rows, cols);
	if(inter2 != dest) MAT_RESHAPE(inter2, rows, cols);

	mat_t *tmp = inter2;
	if(CUR_SCRATCH[3]) { // Swap buffers
		inter2 = inter1;
		inter1 = tmp;
	}
	uint16_t pos = CUR_SCRATCH[0];
	uint16_t idx = CUR_SCRATCH[1];

	uint16_t k = idx / (fcols * frows); // Layers
	uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
	uint16_t n = idx % fcols; // Cols
	uint16_t filter_tile_size = greatest_tile_size(fcols, tile_size);
	if(n + filter_tile_size >= fcols) filter_tile_size = fcols - n;
	uint16_t filter_length = filter_tile_size + (filter_tile_size & 0x01);

	uint16_t common_tile_size = greatest_tile_size(cols, tile_size);
	uint16_t common_rows = tile_size / (cols + filter_length);
	if(common_rows == 0) common_rows = 1;
	else if(common_rows > rows) common_rows = rows;
	uint16_t common_cols = common_tile_size; 
	common_tile_size *= common_rows; 

	// Create filter
	if(!CUR_SCRATCH[2]) {
		if(pos == 0) idx += filter->sparse.offsets[pos];
		uint16_t f = idx % filter_tile_size;
		while(pos < total_elements && 
			f < filter_tile_size) {
			coalesced_filter[f] = MAT_GET(filter, pos);
			pos++;
			f += filter->sparse.offsets[pos];
			idx += filter->sparse.offsets[pos];
		}
		scratch_bak[0] = pos;
		scratch_bak[1] = idx;
		scratch_bak[2] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}

	msp_status status;
	msp_fir_q15_params params_fir = {
		.coeffs = tsrc1,
		.tapLength = filter_length,
	};
	msp_add_q15_params params_add;		

	for(uint16_t i = 0; i < filter_length; i++) {
		if((filter_tile_size & 0x01) && i == filter_length - 1) {
			tsrc1[filter_length - i - 1] = 0;
			continue;
		}
		// tsrc1[filter_length - i - 1] = coalesced_filter[i] << (SHIFT + 1);
		tsrc1[filter_length - i - 1] = coalesced_filter[i] << SHIFT;
	}
	// PRINTF("\r\nFilter ");
	// for(uint16_t i = 0; i < filter_length; i++) {
	// 	PRINTF("%i ", tsrc1[i]);
	// }
	uint16_t row_step = common_rows;
	for(uint16_t i = CUR_SCRATCH[4]; i < rows; 
		i = (CUR_SCRATCH[4] += row_step)) {
		for(uint16_t j = CUR_SCRATCH[5]; j < cols; 
			j = (CUR_SCRATCH[5] += common_tile_size)) {
			params_fir.length = common_tile_size + row_step * filter_length;
			params_fir.length += params_fir.length & 0x01;
			params_add.length = params_fir.length;
			for(uint16_t g = 0; g < row_step; g++) {
				if(common_cols > 12 DMA_ENABLE) { // Load activation tile
					DMA_setTransferSize(dma_config.channelSelect, 
						common_cols);
				    DMA_setSrcAddress(dma_config.channelSelect, 
						(uint32_t) MAT_PTR(src, k, i + l + g, j + n), 
						DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dma_config.channelSelect, 
				    	(uint32_t) (tsrc2 + g * (cols + filter_length)), 
				    	DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dma_config.channelSelect);
				    DMA_startSleepTransfer(dma_config.channelSelect);
				} else {
					memcpy(tsrc2 + g * (cols + filter_length), 
						MAT_PTR(src, k, i + l + g, j + n), 
						sizeof(fixed) * common_cols);	
				}
			}
			status = msp_fir_q15(&params_fir, tsrc2, tdest1);
			msp_checkStatus(status);
			// PRINTF("\r\n i: %u j: %u k: %u l: %u n: %u tsrc1: %i tsrc2: %i tdest1: %i inter: %i row_step: %u  common_cols: %u, common_tile_size: %u",
			// 	i, j, k, l, n, tsrc1[0], tsrc2[0], tdest1[0], MAT_GET(inter1, 0, 0), row_step, common_cols, common_tile_size);
			if(k == 0 && l == 0 && n == 0) { // Zero
				for(uint16_t g = 0; g < row_step; g++) {
					if(common_cols > 12 DMA_ENABLE) {
						DMA_setTransferSize(dma_config.channelSelect, 
							common_cols);
					    DMA_setSrcAddress(dma_config.channelSelect, 
							(uint32_t) (tdest1 + g * (cols + filter_length)), 
							DMA_DIRECTION_INCREMENT);
					    DMA_setDstAddress(dma_config.channelSelect, 
					    	(uint32_t) (MAT_PTR(inter2, i + g, j)), 
					    	DMA_DIRECTION_INCREMENT);
						DMA_enableTransfers(dma_config.channelSelect);
					    DMA_startSleepTransfer(dma_config.channelSelect);
					} else {
						memcpy(MAT_PTR(inter2, i + g, j), 
							tdest1 + g * (cols + filter_length), 
							sizeof(fixed) * common_cols);	
					}
				}
				continue;
			}
			for(uint16_t g = 0; g < row_step; g++) {
				if(common_cols > 12 DMA_ENABLE) { // inter1mediate
					DMA_setTransferSize(dma_config.channelSelect, 
						common_cols);
				    DMA_setSrcAddress(dma_config.channelSelect, 
						(uint32_t) MAT_PTR(inter1, i + g, j), 
						DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dma_config.channelSelect, 
				    	(uint32_t) (tsrc2 + g * (cols + filter_length)), 
				    	DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dma_config.channelSelect);
				    DMA_startSleepTransfer(dma_config.channelSelect);
				} else {
					memcpy(tsrc2 + g * (cols + filter_length), 
						MAT_PTR(inter1, i + g, j), 
						sizeof(fixed) * common_cols);	
				}
			}
			status = msp_add_q15(&params_add, tdest1, tsrc2, tdest2);
			msp_checkStatus(status);
			for(uint16_t g = 0; g < row_step; g++) {
				if(common_cols > 12 DMA_ENABLE) {
					DMA_setTransferSize(dma_config.channelSelect, common_cols);
				    DMA_setSrcAddress(dma_config.channelSelect, 
						(uint32_t) (tdest2 + g * (cols + filter_length)), 
						DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dma_config.channelSelect, 
				    	(uint32_t) (MAT_PTR(inter2, i + g, j)), 
				    	DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dma_config.channelSelect);
				    DMA_startSleepTransfer(dma_config.channelSelect);
				} else {
					memcpy(MAT_PTR(inter2, i + g, j), 
						tdest2 + g * (cols + filter_length), 
						sizeof(fixed) * common_cols);
				}
			}
		}
		if(i + common_rows >= rows) {
			row_step = rows - i;
		}
		CUR_SCRATCH[5] = 0;
	}
	scratch_bak[3] = CUR_SCRATCH[3] ^ 0x01;
	scratch_bak[4] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), 
			(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(scratch_bak[0] < total_elements) {
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), 
			(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
		memset(tsrc1, 0, sizeof(fixed) * filter_tile_size);
		memset(coalesced_filter, 0, sizeof(fixed) * filter_tile_size);
		transition_to(CUR_TASK);
	}

	if(CUR_SCRATCH[3]) {
		for(uint16_t i = CUR_SCRATCH[6]; i < rows; i = (++CUR_SCRATCH[6])){
			for(uint16_t j = CUR_SCRATCH[7]; j < cols; j = (++CUR_SCRATCH[7])){
				MAT_SET(inter1, MAT_GET(inter2, i, j), i, j);
			}
			CUR_SCRATCH[7] = 0;
		}
	}

	if(inter2 != dest) {
		uint16_t i_stride = CUR_SCRATCH[8] / params.stride[1];
		uint16_t j_stride = CUR_SCRATCH[9] / params.stride[2];
		for(uint16_t i = CUR_SCRATCH[8]; i < rows; 
			i = (CUR_SCRATCH[8] += params.stride[1])){
			for(uint16_t j = CUR_SCRATCH[9]; j < cols; 
				j = (CUR_SCRATCH[9] += params.stride[2])){
				MAT_SET(dest, MAT_GET(inter2, i, j), i_stride, j_stride);
				j_stride++;
			}
			i_stride++;
			j_stride = 0;
			CUR_SCRATCH[9] = 0;
		}
	}

	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}