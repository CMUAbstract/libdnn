#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>

#include "nn.h"
#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "fixed.h"
#include "mat.h"

static __fram uint scratch_bak[SCRATCH_SIZE];

void task_cleanup_nonlinear();
TASK(TASK_UID_BLAS_OFFSET + 8, task_cleanup_nonlinear);

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
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint layers = MAT_GET_DIM(src, 0);
	uint rows = MAT_GET_DIM(src, 1);
	uint cols = MAT_GET_DIM(src, 2);

	uint stride = MAT_GET(filter, 0);
	uint size = MAT_GET(filter, 1);

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
	} else if(k + stride[1] == cols) {
		scratch_bak[1] = CUR_INFO.scratch[1] + stride[1];
	}
	scratch_bak[2] = (k + stride[1] == cols) ? 0 : CUR_INFO.scratch[2] + stride[2];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	if(!(CUR_INFO.scratch[0] + 1 == layers &&
		 CUR_INFO.scratch[1] + stride[1] == rows &&
		 CUR_INFO.scratch[2] + stride[2] == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}

void task_relu() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	uint tile_size_x = greatest_tile_size(cols, MAX_TILE_SIZE);
	uint tile_size_y = greatest_tile_size(rows, MAX_TILE_SIZE);
	fixed max = F_LIT(0.0);
	for(uint i = 0; i < tile_size_y; i++) {
		uint idx_i = CUR_INFO.scratch[0] + i;
		for(uint j = 0; j < tile_size_x; j++) {
			uint idx_j = CUR_INFO.scratch[1] + j;
			max = MAT_GET(src, idx_i, idx_j);
			MAT_SET(dest, max, idx_i, idx_j);
			if(F_LT(max, F_LIT(0.0)))
				MAT_SET(dest, F_LIT(0.0), idx_i, idx_j);
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
