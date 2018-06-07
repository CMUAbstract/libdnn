#include <string.h>
#include <msp430.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "nonlinear.h"
#include "blas.h"
#include "mem.h"
#include "state.h"
#include "misc.h"

// Public tasks
TASK(TASK_UID_NONLINEAR_OFFSET + 1, task_pool);
TASK(TASK_UID_NONLINEAR_OFFSET + 2, task_relu);
TASK(TASK_UID_NONLINEAR_OFFSET + 3, task_filter);
TASK(TASK_UID_NONLINEAR_OFFSET + 4, task_transpose);

// Private tasks
void task_cleanup_nonlinear();
TASK(TASK_UID_NONLINEAR_OFFSET + 5, task_cleanup_nonlinear);

// Resets a task
static __fram task_t *last_task;
void task_cleanup_nonlinear() {
	PRINTF("\r\n Cleaning up Nonlinear");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

void task_pool() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t layers = MAT_GET_DIM(src, 0);
	uint16_t rows = MAT_GET_DIM(src, 1);
	for(uint16_t i = CUR_INFO.scratch[0]; i < layers; i = ++CUR_INFO.scratch[0]) {
		for(uint16_t j = CUR_INFO.scratch[1]; j < rows; j = (CUR_INFO.scratch[1] += stride[1])) {
			for(uint16_t k = CUR_INFO.scratch[2]; k < rows; k = (CUR_INFO.scratch[2] += stride[2])) {
				fixed max = MAT_GET(src, i, j, k);
				for(uint16_t l = 0; l < size[1]; l ++) {
					for(uint16_t m = 0; m < size[2]; m ++) {
						fixed val = MAT_GET(src, i, j + l, k + m);
						if(F_LT(max, val))
							max = val;
					}
				}
				MAT_SET(dest, max, i, j / stride[1], k / stride[2]);
			}
			CUR_INFO.scratch[2] = 0;
		}
		CUR_INFO.scratch[1] = 0;
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}

void task_filter() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t layers = MAT_GET_DIM(src, 0);
	uint16_t rows = MAT_GET_DIM(src, 1);
	uint16_t cols = MAT_GET_DIM(src, 2);
	for(uint16_t i = 0; i < layers; i = (CUR_INFO.scratch[0] += stride[0])) {
		for(uint16_t j = 0; j < rows; j = (CUR_INFO.scratch[1] += stride[1])) {
			for(uint16_t k = 0; k < cols; k = (CUR_INFO.scratch[2] += stride[2])) {
				fixed w = MAT_GET(src, i, j, k);
				MAT_SET(dest, w, i / stride[0], j / stride[1], k / stride[2]);
			}
			CUR_INFO.scratch[2] = 0;
		}
		CUR_INFO.scratch[1] = 0;
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}

void task_relu() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t total_elements = MAT_GET_DIM(src, 0) * MAT_GET_DIM(src, 1);
	if(src->len_dims == 3) {
		total_elements *= MAT_GET_DIM(src, 2);
	}
	fixed max = F_LIT(0.0);
	for(uint16_t i = CUR_INFO.scratch[0]; i < total_elements; i = ++CUR_INFO.scratch[0]) {
		max = *(src->data + i);
		*(dest->data + i) = (F_LT(max, F_LIT(0.0))) ? F_LIT(0.0) : max;
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}

void task_transpose() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint16_t j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			MAT_SET(dest, MAT_GET(src, i, j), j, i);
		}
		CUR_INFO.scratch[1] = 0;
	}
	POP_STACK(mat_stack, 2);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nonlinear);
}
