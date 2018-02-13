#include <string.h>
#include <msp430.h>
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
	uint stride = MAT_GET(filter, 0);
	uint size = MAT_GET(filter, 1);
	for(uint i = CUR_INFO.scratch[0]; i < layers; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < rows; j = (CUR_INFO.scratch[1] += stride)) {
			for(uint k = CUR_INFO.scratch[2]; k < rows; k = (CUR_INFO.scratch[2] += stride)) {
				fixed max = MAT_GET(src, i, j, k);
				for(uint l = 0; l < size; l ++) {
					for(uint m = 0; m < size; m ++) {
						fixed val = MAT_GET(src, i, j + l, k + m);
						if(F_LT(max, val))
							max = val;
					}
				}
				MAT_SET(dest, max, i, j / stride, k / stride);
			}
			CUR_INFO.scratch[2] = 0;
		}
		CUR_INFO.scratch[1] = 0;
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
	fixed max = F_LIT(0.0);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			max = MAT_GET(src, i, j);
			MAT_SET(dest, max, i, j);
			if(F_LT(max, F_LIT(0.0)))
				MAT_SET(dest, F_LIT(0.0), i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}
