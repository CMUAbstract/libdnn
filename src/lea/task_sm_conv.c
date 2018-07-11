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
static __fram mat_t buf = {.data = MAT_BUFFER(0)};
static __fram mat_t *buffer = &buf;

// Dense matrix multiplication
void task_sm_conv() {
	uint16_t tile_size = check_calibrate();
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = buffer;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter, rows, cols);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

	mat_t *tmp = dest;
	if(CUR_SCRATCH[3]) { // Swap buffers
		dest = inter;
		inter = tmp;
	}

	// LEA/DMA don't work well for strided convolution
	if(params.stride[1] + params.stride[2] != 2 || fcols == 1) {
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

		uint16_t i_stride = CUR_SCRATCH[3] / params.stride[1];
		uint16_t j_stride = CUR_SCRATCH[4] / params.stride[2];
		for(uint16_t i = CUR_SCRATCH[3]; i < rows * params.stride[1]; 
			i = (CUR_SCRATCH[3] += params.stride[1])){
			for(uint16_t j = CUR_SCRATCH[4]; j < cols * params.stride[2]; 
				j = (CUR_SCRATCH[4] += params.stride[2])){
				fixed w = 0;
				if(!params.same_padding || (i + l < MAT_GET_DIM(src, 1) && 
					j + n < MAT_GET_DIM(src, 2))) {
					w = F_MUL(MAT_GET(filter, pos), 
						(MAT_GET(src, k, i + l, j + n) >> SHIFT));
				}
				if(!zero) {
					w = F_ADD(w, MAT_GET(inter, i_stride, j_stride)); // Zero
				}
				MAT_SET(dest, w, i_stride, j_stride);
				j_stride++;
			}
			j_stride = 0;
			i_stride++;
			CUR_SCRATCH[4] = 0;
		}

		scratch_bak[0] = pos + 1;
		scratch_bak[1] = idx + filter->sparse.offsets[pos + 1];

		scratch_bak[2] = CUR_SCRATCH[2] ^ 0x01;
		scratch_bak[3] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), 
			(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
		if(pos < total_elements - 1) {
			write_to_gbuf((uint8_t *)(scratch_bak + 2), 
				(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
			transition_to(CUR_TASK);
		}
		if(CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[5]; i < rows; i = (++CUR_SCRATCH[5])){
				for(uint16_t j = CUR_SCRATCH[6]; j < cols; j = (++CUR_SCRATCH[6])){
					MAT_SET(inter, MAT_GET(dest, i, j), i, j);
				}
				CUR_SCRATCH[6] = 0;
			}
		}
		POP_STACK(mat_stack, 3);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);	
	}

	/*uint16_t k = CUR_SCRATCH[0];
	uint16_t l = CUR_SCRATCH[1];
	uint16_t n = CUR_SCRATCH[2];
	uint16_t common_tile_size = greatest_tile_size(cols, tile_size);
	uint16_t filter_tile_size = greatest_tile_size(fcols, tile_size);
	if(CUR_SCRATCH[5] + common_tile_size >= cols) 
		common_tile_size = cols - CUR_SCRATCH[5];
	if(n + filter_tile_size >= fcols) 
		filter_tile_size = fcols - n;
	msp_status status;
	uint16_t filter_length = filter_tile_size + (filter_tile_size & 0x01);
	msp_fir_q15_params params_fir = {
		.length = common_tile_size + (common_tile_size & 0x01),
		.coeffs = tsrc1,
		.tapLength = filter_length,
	};
	msp_add_q15_params params_add = {
		.length = common_tile_size + (common_tile_size & 0x01)
	};		

	for(int16_t i = 0; i < filter_length; i++) {
		if((filter_tile_size & 0x01) && i == filter_length - 1) {
			tsrc1[filter_length - i - 1] = 0;
			continue;
		}
		tsrc1[filter_length - i - 1] = MAT_GET(filter, k, l, n + i) << (SHIFT + 1);
	}
	for(uint16_t i = CUR_SCRATCH[4]; i < rows; i = ++CUR_SCRATCH[4]) {
		for(uint16_t j = CUR_SCRATCH[5]; j < cols; j = (CUR_SCRATCH[5] += common_tile_size)) {
			if(common_tile_size > 12 DMA_ENABLE) { // Load activation tile
				DMA_setTransferSize(dma_config.channelSelect, common_tile_size);
			    DMA_setSrcAddress(dma_config.channelSelect, 
					(uint32_t) MAT_PTR(src, k, i + l, j + n), DMA_DIRECTION_INCREMENT);
			    DMA_setDstAddress(dma_config.channelSelect, (uint32_t) (tsrc2),
					DMA_DIRECTION_INCREMENT);
				DMA_enableTransfers(dma_config.channelSelect);
			    DMA_startSleepTransfer(dma_config.channelSelect);
			} else {
				memcpy(tsrc2, MAT_PTR(src, k, i + l, j + n), 
					sizeof(fixed) * common_tile_size);	
			}
			status = msp_fir_q15(&params_fir, tsrc2, tdest1);
			// PRINTF("\r\n status: %u", status);
			msp_checkStatus(status);
			// PRINTF("\r\n i: %u j: %u k: %u l: %u n: %u tsrc1: %i tsrc2: %i tdest1: %i inter: %i",
				// i, j, k, l, n, tsrc1[0], tsrc2[0], tdest1[0], MAT_GET(inter, 0, 0));
			if(k == 0 && l == 0 && n == 0) { // Zero
				if(common_tile_size > 12 DMA_ENABLE) {
					DMA_setTransferSize(dma_config.channelSelect, common_tile_size);
				    DMA_setSrcAddress(dma_config.channelSelect, 
						(uint32_t) (tdest1), DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dma_config.channelSelect, 
				    	(uint32_t) (MAT_PTR(dest, i, j)), DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dma_config.channelSelect);
				    DMA_startSleepTransfer(dma_config.channelSelect);
				} else {
					memcpy(MAT_PTR(dest, i, j), tdest1, 
						sizeof(fixed) * common_tile_size);	
				}
				continue;
			}
			if(common_tile_size > 12 DMA_ENABLE) { // intermediate
				DMA_setTransferSize(dma_config.channelSelect, common_tile_size);
			    DMA_setSrcAddress(dma_config.channelSelect, 
					(uint32_t) MAT_PTR(inter, i, j), DMA_DIRECTION_INCREMENT);
			    DMA_setDstAddress(dma_config.channelSelect, (uint32_t) (tsrc2),
					DMA_DIRECTION_INCREMENT);
				DMA_enableTransfers(dma_config.channelSelect);
			    DMA_startSleepTransfer(dma_config.channelSelect);
			} else {
				memcpy(tsrc2, MAT_PTR(inter, i, j), 
					sizeof(fixed) * common_tile_size);	
			}
			status = msp_add_q15(&params_add, tdest1, tsrc2, tdest2);
			// PRINTF("\r\n status: %u", status);
			msp_checkStatus(status);
			// PRINTF("\r\n tdest2: %i", tdest2[0]);
			if(common_tile_size > 12 DMA_ENABLE) {
				DMA_setTransferSize(dma_config.channelSelect, common_tile_size);
			    DMA_setSrcAddress(dma_config.channelSelect, 
					(uint32_t) (tdest2), DMA_DIRECTION_INCREMENT);
			    DMA_setDstAddress(dma_config.channelSelect, 
			    	(uint32_t) (MAT_PTR(dest, i, j)), DMA_DIRECTION_INCREMENT);
				DMA_enableTransfers(dma_config.channelSelect);
			    DMA_startSleepTransfer(dma_config.channelSelect);
			} else {
				memcpy(MAT_PTR(dest, i, j), tdest2, 
					sizeof(fixed) * common_tile_size);
				// PRINTF("\r\n %i dest: %i inter: %i", 
					// tdest2[0], MAT_GET(dest, 0, 0), MAT_GET(inter, 0, 0));

			}
		}
		CUR_SCRATCH[5] = 0;
		common_tile_size = greatest_tile_size(cols, tile_size);
	}

	scratch_bak[0] = k;
	scratch_bak[1] = l;
	scratch_bak[2] = n + filter_tile_size;
	if(n + filter_tile_size >= fcols && l + 1 >= frows) {
		scratch_bak[0] = k + 1;
		scratch_bak[1] = 0;
		scratch_bak[2] = 0;
	}else if(n + filter_tile_size >= fcols) {
		scratch_bak[1] = l + 1;
		scratch_bak[2] = 0;
	}

	scratch_bak[3] = CUR_SCRATCH[3] ^ 0x01;	
	scratch_bak[4] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), 
		(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(!(k + 1 >= flayers && l + 1 >= frows && n + common_tile_size >= fcols)) {
		write_to_gbuf((uint8_t *)(scratch_bak + 3), 
			(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
		memset(tsrc1, 0, sizeof(fixed) * greatest_tile_size(fcols, tile_size));
		memset(tsrc2, 0, sizeof(fixed) * common_tile_size);
		memset(tdest1, 0, sizeof(fixed) * common_tile_size);
		transition_to(CUR_TASK);
	}*/

	if(CUR_SCRATCH[3]) {
		for(uint16_t i = CUR_SCRATCH[6]; i < rows; i = (++CUR_SCRATCH[6])){
			for(uint16_t j = CUR_SCRATCH[7]; j < cols; j = (++CUR_SCRATCH[7])){
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_SCRATCH[7] = 0;
		}
	}

	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}