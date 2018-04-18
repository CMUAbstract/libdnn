#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>

#include "nonlinear.h"
#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "fixed.h"
#include "mat.h"
#include "misc.h"

// Public tasks
TASK(TASK_UID_NONLINEAR_OFFSET + 1, task_pool);
TASK(TASK_UID_NONLINEAR_OFFSET + 2, task_relu);
TASK(TASK_UID_NONLINEAR_OFFSET + 3, task_filter);
TASK(TASK_UID_NONLINEAR_OFFSET + 4, task_transpose);

// Private tasks
void task_cleanup_nonlinear();
TASK(TASK_UID_NONLINEAR_OFFSET + 5, task_cleanup_nonlinear);

static __fram uint scratch_bak[SCRATCH_SIZE];

// Resets a task
static __fram task_t *last_task;
void task_cleanup_nonlinear() {
	PRINTF("\r\n Cleaning up NN");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

void task_pool() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);

	uint layers = MAT_GET_DIM(src, 0);
	uint rows = MAT_GET_DIM(src, 1);
	uint cols = MAT_GET_DIM(src, 2);

	uint i = CUR_INFO.scratch[0];
	uint j = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];
	fixed max = MAT_GET(src, i, j, k);
	for(uint l = 0; l < size[1]; l ++) {
		for(uint m = 0; m < size[2]; m ++) {
			fixed val = MAT_GET(src, i, j + l, k + m);
			if(F_LT(max, val))
				max = val;
		}
	}
	MAT_SET(dest, max, i, j / stride[1], k / stride[2]);
	scratch_bak[0] = CUR_INFO.scratch[0];
	scratch_bak[1] = CUR_INFO.scratch[1];
	if(j + stride[1] == rows && k + stride[2] == cols) {
		scratch_bak[0] = CUR_INFO.scratch[0] + 1;
		scratch_bak[1] = 0;
	} else if(k + stride[2] == cols) {
		scratch_bak[1] = CUR_INFO.scratch[1] + stride[1];
	}
	scratch_bak[2] = (k + stride[2] == cols) ? 0 : CUR_INFO.scratch[2] + stride[2];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	if(!(CUR_INFO.scratch[0] + 1 == layers &&
		 CUR_INFO.scratch[1] + stride[1] == rows &&
		 CUR_INFO.scratch[2] + stride[2] == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}

void task_filter() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);

	uint layers = MAT_GET_DIM(src, 0);
	uint rows = MAT_GET_DIM(src, 1);
	uint cols = MAT_GET_DIM(src, 2);

	uint i = CUR_INFO.scratch[0];
	uint j = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];

	fixed w = MAT_GET(src, i, j, k);
	MAT_SET(dest, w, i / stride[0], j / stride[1], k / stride[2]);
	scratch_bak[0] = CUR_INFO.scratch[0];
	scratch_bak[1] = CUR_INFO.scratch[1];
	if(j + stride[1] == rows && k + stride[2] == cols) {
		scratch_bak[0] = CUR_INFO.scratch[0] + stride[0];
		scratch_bak[1] = 0;
	} else if(k + stride[1] == cols) {
		scratch_bak[1] = CUR_INFO.scratch[1] + stride[1];
	}
	scratch_bak[2] = (k + stride[1] == cols) ? 0 : CUR_INFO.scratch[2] + stride[2];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	if(!(CUR_INFO.scratch[0] + stride[0] == layers &&
		 CUR_INFO.scratch[1] + stride[1] == rows &&
		 CUR_INFO.scratch[2] + stride[2] == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}

void task_relu() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint total_elements = MAT_GET_DIM(src, 0) * MAT_GET_DIM(src, 1);
	if(src->len_dims == 3) {
		total_elements *= MAT_GET_DIM(src, 2);
	}
	fixed max = F_LIT(0.0);
	uint tile_size = greatest_tile_size(total_elements, CONFIG_TILE_SIZE);
	for(uint i = 0; i < tile_size; i++) {
		uint idx_i = CUR_INFO.scratch[0] + i;
		max = *(src->data + idx_i);
		*(dest->data + idx_i) = (F_LT(max, F_LIT(0.0))) ? F_LIT(0.0) : max;
	}
	scratch_bak[0] = CUR_INFO.scratch[0] + tile_size;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	if(!(CUR_INFO.scratch[0] + tile_size == total_elements)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}

void task_transpose() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	uint tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint i = 0; i < tile_size_y; i++) {
		uint idx_i = CUR_INFO.scratch[0] + i;
		for(uint j = 0; j < tile_size_x; j++) {
			uint idx_j = CUR_INFO.scratch[1] + j;
			fixed val = MAT_GET(src, idx_i, idx_j);
			MAT_SET(dest, val, idx_j, idx_i);
		}
	}
	scratch_bak[0] = (CUR_INFO.scratch[1] + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : 0;
	scratch_bak[1] = (CUR_INFO.scratch[1] + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows &&
		 CUR_INFO.scratch[1] + tile_size_x == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}
