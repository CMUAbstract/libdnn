#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "blas.h"
#include "mem.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"
#include "profile.h"
#include "cleanup.h"
#include "tile.h"

TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv);

static __fram mat_t buf = {.data = MAT_BUFFER(0)};
static __fram mat_t *inter = &buf;

void task_sm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	uint16_t tile_size_x = greatest_tile_size(
		params.stride[2] * cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(
		params.stride[1] * rows, CONFIG_TILE_SIZE);
	MAT_RESHAPE(inter, rows, cols);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

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

	uint16_t init_i = CUR_SCRATCH[2];
	if(init_i % params.stride[1]) 
		init_i += (params.stride[1] - (tile_size_y % params.stride[1]));
	uint16_t init_j = CUR_SCRATCH[3];
	if(init_j % params.stride[2]) 
		init_j += (params.stride[2] - (tile_size_x % params.stride[2]));
	uint16_t i_stride = init_i / params.stride[1];
	uint16_t j_stride = init_j / params.stride[2];
	prof_pulse(0x1);
	for(uint16_t i = init_i; 
		i < CUR_SCRATCH[2] + tile_size_y; i += params.stride[1]){
		for(uint16_t j = init_j; 
			j < CUR_SCRATCH[3] + tile_size_x; j += params.stride[2]){
			fixed w = 0;
			if(!params.same_padding || (i + l < MAT_GET_DIM(src, 1) && 
				j + n < MAT_GET_DIM(src, 2))) {
				w = F_MUL(MAT_GET(filter, pos), 
					MAT_GET(src, k, i + l, j + n));
			}
			if(!zero) {
				w = F_ADD(w, MAT_GET(dest, i_stride, j_stride)); // Zero
			}
			MAT_SET(inter, w, i_stride, j_stride);
			write_to_gbuf((uint8_t *)(inter->data + cols * i_stride + j_stride), 
					(uint8_t *)(dest->data + cols * i_stride + j_stride), sizeof(fixed));	
			j_stride++;
		}
		j_stride = init_j / params.stride[2];
		i_stride++;
	}
	prof_pulse(0x1);

	scratch_bak[0] = pos;
	scratch_bak[1] = idx;
	scratch_bak[2] = CUR_SCRATCH[2];
	scratch_bak[3] = CUR_SCRATCH[3] + tile_size_x;
	if(CUR_SCRATCH[2] + tile_size_y >= rows * params.stride[1] && 
		CUR_SCRATCH[3] + tile_size_x >= cols * params.stride[2]) {
		scratch_bak[0] = pos + 1;
		scratch_bak[1] = idx + filter->sparse.offsets[pos + 1];
		scratch_bak[2] = 0;
		scratch_bak[3] = 0;
	} else if(CUR_SCRATCH[3] + tile_size_x >= cols * params.stride[2]) {
		scratch_bak[2] = CUR_SCRATCH[2] + tile_size_y;
		scratch_bak[3] = 0;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), 
		(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	if(pos < total_elements - 1) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
