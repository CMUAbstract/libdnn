#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "nonlinear.h"
#include "blas.h"
#include "mem.h"
#include "state.h"
#include "misc.h"
#include "cleanup.h"
#include "tile.h"

// Public tasks
TASK(TASK_UID_NONLINEAR_OFFSET + 1, task_pool);
TASK(TASK_UID_NONLINEAR_OFFSET + 2, task_relu);
TASK(TASK_UID_NONLINEAR_OFFSET + 3, task_filter);
TASK(TASK_UID_NONLINEAR_OFFSET + 4, task_transpose);

void task_pool() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);

	uint16_t layers = MAT_GET_DIM(src, 0);
	uint16_t rows = MAT_GET_DIM(src, 1);
	uint16_t cols = MAT_GET_DIM(src, 2);

	uint16_t i = CUR_SCRATCH[0];
	uint16_t j = CUR_SCRATCH[1];
	uint16_t k = CUR_SCRATCH[2];
	fixed max = MAT_GET(src, i, j, k);
	for(uint16_t l = 0; l < params.size[1]; l ++) {
		for(uint16_t m = 0; m < params.size[2]; m ++) {
			fixed val = MAT_GET(src, i, j + l, k + m);
			if(F_LT(max, val))
				max = val;
		}
	}
	MAT_SET(dest, max, i, j / params.stride[1], k / params.stride[2]);
	scratch_bak[0] = CUR_SCRATCH[0];
	scratch_bak[1] = CUR_SCRATCH[1];
	if(j + params.stride[1] == rows && k + params.stride[2] == cols) {
		scratch_bak[0] = CUR_SCRATCH[0] + 1;
		scratch_bak[1] = 0;
	} else if(k + params.stride[2] == cols) {
		scratch_bak[1] = CUR_SCRATCH[1] + params.stride[1];
	}
	scratch_bak[2] = (k + params.stride[2] == cols) ? 0 : CUR_SCRATCH[2] + params.stride[2];
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	if(!(CUR_SCRATCH[0] + 1 == layers &&
		 CUR_SCRATCH[1] + params.stride[1] == rows &&
		 CUR_SCRATCH[2] + params.stride[2] == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_filter() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);

	uint16_t layers = MAT_GET_DIM(src, 0);
	uint16_t rows = MAT_GET_DIM(src, 1);
	uint16_t cols = MAT_GET_DIM(src, 2);

	uint16_t i = CUR_SCRATCH[0];
	uint16_t j = CUR_SCRATCH[1];
	uint16_t k = CUR_SCRATCH[2];

	fixed w = MAT_GET(src, i, j, k);
	MAT_SET(dest, w, i / params.stride[0], j / params.stride[1], k / params.stride[2]);
	scratch_bak[0] = CUR_SCRATCH[0];
	scratch_bak[1] = CUR_SCRATCH[1];
	if(j + params.stride[1] == rows && k + params.stride[2] == cols) {
		scratch_bak[0] = CUR_SCRATCH[0] + params.stride[0];
		scratch_bak[1] = 0;
	} else if(k + params.stride[1] == cols) {
		scratch_bak[1] = CUR_SCRATCH[1] + params.stride[1];
	}
	scratch_bak[2] = (k + params.stride[1] == cols) ? 0 : CUR_SCRATCH[2] + params.stride[2];
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	if(!(CUR_SCRATCH[0] + params.stride[0] == layers &&
		 CUR_SCRATCH[1] + params.stride[1] == rows &&
		 CUR_SCRATCH[2] + params.stride[2] == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_relu() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t total_elements = MAT_GET_DIM(src, 0) * MAT_GET_DIM(src, 1);
	if(src->len_dims == 3) {
		total_elements *= MAT_GET_DIM(src, 2);
	}
	fixed max = F_LIT(0.0);
	uint16_t tile_size = greatest_tile_size(total_elements, CONFIG_TILE_SIZE);
	for(uint16_t i = 0; i < tile_size; i++) {
		uint16_t idx_i = CUR_SCRATCH[0] + i;
		max = *(src->data + idx_i);
		*(dest->data + idx_i) = (F_LT(max, F_LIT(0.0))) ? F_LIT(0.0) : max;
	}
	scratch_bak[0] = CUR_SCRATCH[0] + tile_size;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	if(!(CUR_SCRATCH[0] + tile_size == total_elements)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_transpose() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint16_t i = 0; i < tile_size_y; i++) {
		uint16_t idx_i = CUR_SCRATCH[0] + i;
		for(uint16_t j = 0; j < tile_size_x; j++) {
			uint16_t idx_j = CUR_SCRATCH[1] + j;
			fixed val = MAT_GET(src, idx_i, idx_j);
			MAT_SET(dest, val, idx_j, idx_i);
		}
	}
	scratch_bak[0] = 0;
	scratch_bak[1] = CUR_SCRATCH[1] + tile_size_x;
	if(CUR_SCRATCH[1] + tile_size_x == cols) {
		scratch_bak[0] = CUR_SCRATCH[0] + tile_size_y;
		scratch_bak[1] = 0;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	if(!(CUR_SCRATCH[0] + tile_size_y == rows &&
		 CUR_SCRATCH[1] + tile_size_x == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
