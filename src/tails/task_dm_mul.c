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
#include "cleanup.h"

TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_mul);

static __fram mat_t buf = {.data = LAYER_BUFFER(3)};
static __fram mat_t src_copy;
static __fram mat_t *buffer = &buf;

// Dense matrix multiplication
void task_dm_mul() {
	uint16_t tile_size = check_calibrate();
	mat_t *src = &src_copy;
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = buffer;
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(filter, 0);
	uint16_t cols = MAT_GET_DIM(filter, 1);
	uint16_t dcols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter, rows, cols);

	uint16_t k = CUR_SCRATCH[2];
	uint16_t common_tile_size = greatest_tile_size(cols, tile_size);
	if(k + common_tile_size >= cols) common_tile_size = cols - k;
	msp_mac_q15_params params = {
		.length = common_tile_size + (common_tile_size & 0x01)
	};
	msp_status status;

	// Run once
	if(!CUR_SCRATCH[4]) {
		MAT_COPY(PEEK_STACK(mat_stack, 0), src);
		MAT_TRANSPOSE(src);
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 4), 
			(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(CUR_SCRATCH[3]) { // Swap buffers
		dest = inter;
		inter = tmp;
	}

	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		if(common_tile_size > 12 DMA_ENABLE) { // Load filter tile
			DMA_setTransferSize(dma_config.channelSelect, common_tile_size);
		    DMA_setSrcAddress(dma_config.channelSelect, 
		    	(uint32_t) MAT_PTR(filter, i, k), DMA_DIRECTION_INCREMENT);
		    DMA_setDstAddress(dma_config.channelSelect, (uint32_t) (tsrc1),
				DMA_DIRECTION_INCREMENT);
		    DMA_enableTransfers(dma_config.channelSelect);
		    DMA_startSleepTransfer(dma_config.channelSelect);
		} else {
			memcpy(tsrc1, MAT_PTR(filter, i, k), sizeof(fixed) * common_tile_size);	
		}
		for(uint16_t j = CUR_SCRATCH[1]; j < dcols; j = ++CUR_SCRATCH[1]) {
			if(common_tile_size > 12 DMA_ENABLE) { // Load activation tile
				DMA_setTransferSize(dma_config.channelSelect, common_tile_size);
			    DMA_setSrcAddress(dma_config.channelSelect, 
					(uint32_t) MAT_PTR(src, k, j), DMA_DIRECTION_INCREMENT);
			    DMA_setDstAddress(dma_config.channelSelect, (uint32_t) (tsrc2),
					DMA_DIRECTION_INCREMENT);
				DMA_enableTransfers(dma_config.channelSelect);
			    DMA_startSleepTransfer(dma_config.channelSelect);
			} else {
				memcpy(tsrc2, MAT_PTR(src, k, j), 
					sizeof(fixed) * common_tile_size);
			}
			// Do dot product here
			status = msp_mac_q15(&params, tsrc1, tsrc2, tdest1);
			msp_checkStatus(status);
			fixed w = ((*tdest1 >> 1) + F_K) >> F_N;
			// fixed w = *tdest1 >> 1;
			// PRINTF("\r\n i: %u j: %u k: %u filter: %i src: %i tsrc1: %i tsrc2: %i dest: %i", 
				// i, j, k, *MAT_PTR(filter, i, k), *MAT_PTR(src, k, j), tsrc1[0], tsrc2[0], w);
			if(k > 0) {
				w = F_ADD(w, MAT_GET(dest, i, j));
			}
			MAT_SET(dest, w, i, j);
		}
		CUR_SCRATCH[1] = 0;
	}

	scratch_bak[0] = 0;
	// Inc by tile size
	scratch_bak[2] = k + common_tile_size;
	scratch_bak[3] = CUR_SCRATCH[3] ^ 0x01;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	if(k + common_tile_size < cols) {
		write_to_gbuf((uint8_t *)(scratch_bak + 3), 
			(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
		memset(tsrc1, 0, sizeof(fixed) * common_tile_size);
		transition_to(CUR_TASK);
	}
	if(CUR_SCRATCH[3]) {
		for(uint16_t i = CUR_SCRATCH[5]; i < rows; i = ++CUR_SCRATCH[5]) {
			for(uint16_t j = CUR_SCRATCH[6]; j < dcols; j = ++CUR_SCRATCH[6]) {
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_SCRATCH[1] = 0;
		}
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}