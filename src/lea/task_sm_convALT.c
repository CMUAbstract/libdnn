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

#define FILTER_ROWS 0x8
static __fram fixed coalesced_filter1[FILTER_ROWS][CONFIG_TILE_SIZE];
static __fram fixed coalesced_filter2[FILTER_ROWS][CONFIG_TILE_SIZE];
static __fram bool row_mask[FILTER_ROWS];
static __fram uint16_t row_i;

// Dense matrix multiplication
void task_sm_conv() {
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter1 = buffer1;
	mat_t *inter2 = buffer2;
	uint16_t tile_size = 0;
	uint16_t flayers = filter->sparse.dims[0];
	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t total_elements = MAT_GET_DIM(filter, 0);
	bool run_sonic = false; // Run Sonic?
	if(!run_sonic) {
		tile_size = check_calibrate();
	}

	if(run_sonic) {
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

	prof_pulse(0x10);
	if(params.transpose) {
		uint16_t tmp = rows;
		rows = cols;
		cols = tmp;
		tmp = frows;
		frows = fcols;
		fcols = tmp;
	}
	rows *= params.stride[1];
	cols *= params.stride[2];
	uint16_t srows = MAT_GET_DIM(src, 1);
	uint16_t scols = MAT_GET_DIM(src, 2);
	uint16_t drows = rows;
	uint16_t dcols = cols;

	// PRINTF("\r\n rows: %u cols: %u", rows, cols);
	MAT_RESHAPE(inter1, drows, dcols);
	MAT_RESHAPE(inter2, drows, dcols);

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
	uint16_t filter_cols = greatest_tile_size(fcols, CONFIG_TILE_SIZE);
	uint16_t filter_rows = (CONFIG_TILE_SIZE * FILTER_ROWS) / fcols;
	if(filter_rows > FILTER_ROWS) filter_rows = FILTER_ROWS;
	if(filter_rows == 0) filter_rows = 0;
	else if(filter_rows > frows) filter_rows = frows;
	else if(filter_rows + l > frows) filter_rows = frows - l;
	uint16_t filter_length = filter_cols + (filter_cols & 0x01);

	uint16_t common_tile_size = greatest_tile_size(scols, tile_size);
	uint16_t common_rows = tile_size / (scols + filter_length);
	if(common_rows == 0) common_rows = 1;
	else if(common_rows > rows) common_rows = srows;
	uint16_t common_cols = common_tile_size; 
	uint16_t dcommon_cols = common_cols;
	if(dcommon_cols > dcols) dcommon_cols = dcols;
	common_tile_size *= common_rows;
	prof_pulse(0x10);

PRINTF("\r\n rows: %u cols: %u common_rows: %u common_cols: %u filter_rows: %u filter_cols: %u", 
rows, cols, common_rows, common_cols, filter_rows, filter_cols);

	// Create filter
	if(CUR_SCRATCH[2] == 0) {
		prof_pulse(0x30);
		memset(row_mask, false, sizeof(bool) * FILTER_ROWS);
		if(pos == 0) idx += filter->sparse.offsets[pos];
		uint16_t input_size = fcols * frows;
		k = idx / input_size;
		// PRINTF("\r\n");
		row_i = 0;
		for(uint16_t i = l; 
			i < l + filter_rows && pos < total_elements && k == idx / input_size; 
			i = (idx % input_size) / fcols) {
			uint16_t f = idx % fcols;
			while(pos < total_elements && f < filter_cols) {
				row_mask[row_i] = true;
				coalesced_filter1[row_i][f] = MAT_GET(filter, pos);
				pos++;
				f += filter->sparse.offsets[pos];
				idx += filter->sparse.offsets[pos];
				// PRINTF("(%u, %u, %u, %u) ", pos, idx, i, f);
			}
			row_i++;
		}
		scratch_bak[0] = pos;
		scratch_bak[1] = idx;
		scratch_bak[2] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		prof_pulse(0x30);
		transition_to(CUR_TASK);
	} else if(CUR_SCRATCH[2] == 1) {
		prof_pulse(0x40);
		for(uint16_t i = CUR_SCRATCH[4]; 
			i < row_i; i = ++CUR_SCRATCH[4]) {
			for(uint16_t j = CUR_SCRATCH[5]; 
				j < filter_length; j = ++CUR_SCRATCH[5]) {
				if((filter_cols & 0x01) && j == filter_length - 1) {
					coalesced_filter2[i][filter_length - j - 1] = 0;
					continue;
				}
				if(dcols == 1) {
					coalesced_filter2[i][filter_length - j - 1] = coalesced_filter1[i][j];
					continue;
				}
				// coalesced_filter2[i][filter_length - j - 1] = 
					// coalesced_filter1[i][j] << (SHIFT + 1);
				coalesced_filter2[i][filter_length - j - 1] = 
					coalesced_filter1[i][j] << SHIFT;
			}
			CUR_SCRATCH[5] = 0;
		}
		scratch_bak[2] = 2;
		scratch_bak[4] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), 
			(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
		prof_pulse(0x40);
		transition_to(CUR_TASK);
	}

	msp_status status;
	msp_fir_q15_params params_fir = {
		.coeffs = tsrc1,
		.tapLength = filter_length,
	};
	msp_add_q15_params params_add;	
	msp_mac_q15_params params_mac;	

	// PRINTF("\r\nFilter ");
	// for(uint16_t i = 0; i < filter_rows; i++) {
	// 	PRINTF("\r\n");
	// 	for(uint16_t j = 0; j < filter_length; j++) {
	// 		PRINTF("%i ", coalesced_filter2[i][j]);
	// 	}
	// }
	uint16_t row_step = common_rows;
	prof_pulse(0x50);
	for(uint16_t i = CUR_SCRATCH[4]; i < drows; 
		i = (CUR_SCRATCH[4] += row_step)) {
		for(uint16_t j = CUR_SCRATCH[5]; j < dcols; 
			j = (CUR_SCRATCH[5] += dcommon_cols)) {
			params_fir.length = common_tile_size + row_step * filter_length;
			params_fir.length += params_fir.length & 0x01;
			params_add.length = params_fir.length;
			uint16_t filter_idx = 0;
			bool loaded_partial = false;
			bool swap_lea_bufs = false;
			uint16_t orig_l = l;
			for(uint16_t l = orig_l; l < orig_l + filter_rows; l++) {
				if(!row_mask[filter_idx]) continue;
				if(filter_length > 12 DMA_ENABLE) { // Load activation tile
					DMA_setTransferSize(dma_config.channelSelect, 
						filter_length);
				    DMA_setSrcAddress(dma_config.channelSelect, 
						(uint32_t) (coalesced_filter2[filter_idx++]), 
						DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dma_config.channelSelect, 
				    	(uint32_t) (tsrc1), 
				    	DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dma_config.channelSelect);
				    DMA_startSleepTransfer(dma_config.channelSelect);
				} else {
					memcpy(tsrc1, coalesced_filter2[filter_idx++], 
						sizeof(fixed) * filter_length);	
				}
				// common_cols should be based on source dimensions
				for(uint16_t g = 0; g < row_step; g++) {
					if(common_cols > 12 DMA_ENABLE) { // Load activation tile
						DMA_setTransferSize(dma_config.channelSelect, 
							common_cols);
					    DMA_setSrcAddress(dma_config.channelSelect, 
							(uint32_t) MAT_PTR(src, k, i + l + g, j + n), 
							DMA_DIRECTION_INCREMENT);
					    DMA_setDstAddress(dma_config.channelSelect, 
					    	(uint32_t) (tsrc2 + g * (common_cols + filter_length)), 
					    	DMA_DIRECTION_INCREMENT);
						DMA_enableTransfers(dma_config.channelSelect);
					    DMA_startSleepTransfer(dma_config.channelSelect);
					} else {
						memcpy(tsrc2 + g * (common_cols + filter_length), 
							MAT_PTR(src, k, i + l + g, j + n), 
							sizeof(fixed) * common_cols);	
					}
				}
				if(dcols == 1) {
					params_mac.length = common_cols;
					for(uint16_t g = 0; g < row_step; g++) {
						uint16_t ptr_offset = g * (common_cols + filter_length);
						fixed *tdest = tdest1 + ptr_offset;
						status = msp_mac_q15(&params_mac, tsrc1, 
							tsrc2 + ptr_offset, tdest);
						*tdest = ((*tdest >> 1) + F_K) >> F_N;
					}
				} else {
					status = msp_fir_q15(&params_fir, tsrc2, tdest1);
					msp_checkStatus(status);
				}
// PRINTF("\r\n i: %u j: %u k: %u l: %u n: %u", i, j, k, l, n);
				if(k == 0 && l == 0 && n == 0 && filter_rows == 1) { // Zero
					// Do a move from tdest1 to tdest2
					if(params_add.length > 12 DMA_ENABLE) {
						DMA_setTransferSize(dma_config.channelSelect, 
							params_add.length);
					    DMA_setSrcAddress(dma_config.channelSelect, 
							(uint32_t) (tdest1), 
							DMA_DIRECTION_INCREMENT);
					    DMA_setDstAddress(dma_config.channelSelect, 
					    	(uint32_t) (tdest2), 
					    	DMA_DIRECTION_INCREMENT);
						DMA_enableTransfers(dma_config.channelSelect);
					    DMA_startSleepTransfer(dma_config.channelSelect);
					} else {
						memcpy(tdest2, tdest1, sizeof(fixed) * params_add.length);	
					}
					swap_lea_bufs = true;
					continue;
				} else if(k == 0 && l == 0 && n == 0 && !loaded_partial) {
					// Do a move from tdest1 to tsrc3
					if(params_add.length > 12 DMA_ENABLE) {
						DMA_setTransferSize(dma_config.channelSelect, 
							params_add.length);
					    DMA_setSrcAddress(dma_config.channelSelect, 
							(uint32_t) (tdest1), 
							DMA_DIRECTION_INCREMENT);
					    DMA_setDstAddress(dma_config.channelSelect, 
					    	(uint32_t) (tsrc3), 
					    	DMA_DIRECTION_INCREMENT);
						DMA_enableTransfers(dma_config.channelSelect);
					    DMA_startSleepTransfer(dma_config.channelSelect);
					} else {
						memcpy(tsrc3, tdest1, sizeof(fixed) * params_add.length);	
					}
					loaded_partial = true;
					continue;
				} else if(!loaded_partial) {
					// Load in partial from memory into tsrc3
					for(uint16_t g = 0; g < row_step; g++) {
						if(common_cols > 12 DMA_ENABLE) { // intermediate
							DMA_setTransferSize(dma_config.channelSelect, 
								dcommon_cols);
						    DMA_setSrcAddress(dma_config.channelSelect, 
								(uint32_t) MAT_PTR(inter1, i + g, j), 
								DMA_DIRECTION_INCREMENT);
						    DMA_setDstAddress(dma_config.channelSelect, 
						    	(uint32_t) (tsrc3 + g * (common_cols + filter_length)), 
						    	DMA_DIRECTION_INCREMENT);
							DMA_enableTransfers(dma_config.channelSelect);
						    DMA_startSleepTransfer(dma_config.channelSelect);
						} else {
							memcpy(tsrc3 + g * (common_cols + filter_length), 
								MAT_PTR(inter1, i + g, j), 
								sizeof(fixed) * dcommon_cols);	
						}
					}
					loaded_partial = true;
				}
				if(swap_lea_bufs) {
					status = msp_add_q15(&params_add, tdest1, tdest2, tsrc3);
					swap_lea_bufs = false;
				} else {
					status = msp_add_q15(&params_add, tdest1, tsrc3, tdest2);
					swap_lea_bufs = true;
				}
				msp_checkStatus(status);
			}
			fixed *tdest = tsrc3;
			if(swap_lea_bufs) {
				tdest = tdest2;
			}
			for(uint16_t g = 0; g < row_step; g++) {
				if(common_cols > 12 DMA_ENABLE) {
					DMA_setTransferSize(dma_config.channelSelect, dcommon_cols);
				    DMA_setSrcAddress(dma_config.channelSelect, 
						(uint32_t) (tdest + g * (common_cols + filter_length)), 
						DMA_DIRECTION_INCREMENT);
				    DMA_setDstAddress(dma_config.channelSelect, 
				    	(uint32_t) (MAT_PTR(inter2, i + g, j)), 
				    	DMA_DIRECTION_INCREMENT);
					DMA_enableTransfers(dma_config.channelSelect);
				    DMA_startSleepTransfer(dma_config.channelSelect);
				} else {
					memcpy(MAT_PTR(inter2, i + g, j), 
						tdest + g * (common_cols + filter_length), 
						sizeof(fixed) * dcommon_cols);
				}
			}
		}
		if(i + common_rows >= rows) {
			row_step = drows - i;
		}
		CUR_SCRATCH[5] = 0;
	}
	prof_pulse(0x50);
	scratch_bak[3] = CUR_SCRATCH[3] ^ 0x01;
	scratch_bak[4] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), 
			(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	memset(tsrc1, 0, sizeof(fixed) * filter_length * filter_rows);
	memset(coalesced_filter1, 0, sizeof(fixed) * filter_length * filter_rows);
	if(scratch_bak[0] < total_elements) {
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), 
			(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
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
	uint16_t i_stride = CUR_SCRATCH[8] / params.stride[1];
	uint16_t j_stride = CUR_SCRATCH[9] / params.stride[2];
	for(uint16_t i = CUR_SCRATCH[8]; i < rows; 
		i = (CUR_SCRATCH[8] += params.stride[1])){
		for(uint16_t j = CUR_SCRATCH[9]; j < cols; 
			j = (CUR_SCRATCH[9] += params.stride[2])){
			if(params.transpose) {
				MAT_SET(dest, MAT_GET(inter2, i, j), j_stride, i_stride);
			} else {
				MAT_SET(dest, MAT_GET(inter2, i, j), i_stride, j_stride);
			}
			j_stride++;
		}
		i_stride++;
		j_stride = 0;
		CUR_SCRATCH[9] = 0;
	}

	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}