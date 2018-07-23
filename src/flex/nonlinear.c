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
#include "cleanup.h"
#include "profile.h"

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
	for(uint16_t i = CUR_SCRATCH[0]; i < layers; i = ++CUR_SCRATCH[0]) {
		prof_inc("loop_inc", 1, 1);
		for(uint16_t j = CUR_SCRATCH[1]; j < rows;
			j = (CUR_SCRATCH[1] += params.stride[1])) {
			prof_inc("loop_inc", 1, 1);
			for(uint16_t k = CUR_SCRATCH[2]; k < rows; 
				k = (CUR_SCRATCH[2] += params.stride[2])) {
				prof_inc("ld", 1, 1);
				prof_inc("loop_inc", 1, 1);
				fixed max = MAT_GET(src, i, j, k);
				for(uint16_t l = 0; l < params.size[1]; l++) {
					prof_inc("inc", 1, 1);
					prof_inc("ld", 1, 1);
					for(uint16_t m = 0; m < params.size[2]; m++) {
						prof_inc("inc", 1, 1);
						prof_inc("MAT_GET_3D", 1, 1);
						prof_inc("add", 2, 2);
						fixed val = MAT_GET(src, i, j + l, k + m);
						if(F_LT(max, val))
							max = val;
					}
				}
				prof_inc("MAT_SET_3D", 1, 1);
				prof_inc("mul", 2, 2);
				MAT_SET(dest, max, i, j / params.stride[1], k / params.stride[2]);
			}
			CUR_SCRATCH[2] = 0;
		}
		CUR_SCRATCH[1] = 0;
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
	for(uint16_t i = 0; i < layers; i = (CUR_SCRATCH[0] += params.stride[0])) {
		for(uint16_t j = 0; j < rows; j = (CUR_SCRATCH[1] += params.stride[1])) {
			for(uint16_t k = 0; k < cols; k = (CUR_SCRATCH[2] += params.stride[2])) {
				fixed w = MAT_GET(src, i, j, k);
				MAT_SET(dest, w, i / params.stride[0], 
					j / params.stride[1], k / params.stride[2]);
			}
			CUR_SCRATCH[2] = 0;
		}
		CUR_SCRATCH[1] = 0;
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
	for(uint16_t i = CUR_SCRATCH[0]; i < total_elements; i = ++CUR_SCRATCH[0]) {
		prof_inc("loop_inc", 1, 1);
		max = *(src->data + i);
		prof_inc("add", 2, 2);
		prof_inc("ld", 1, 1);
		prof_inc("st", 1, 1);
		*(dest->data + i) = (F_LT(max, F_LIT(0.0))) ? F_LIT(0.0) : max;
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
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
			MAT_SET(dest, MAT_GET(src, i, j), j, i);
		}
		CUR_SCRATCH[1] = 0;
	}
	POP_STACK(mat_stack, 2);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
