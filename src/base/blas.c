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

// Public tasks
TASK(TASK_UID_BLAS_OFFSET, task_ds_zero);
TASK(TASK_UID_BLAS_OFFSET + 1, task_ds_add);
TASK(TASK_UID_BLAS_OFFSET + 2, task_ds_mul);
TASK(TASK_UID_BLAS_OFFSET + 3, task_ds_div);
TASK(TASK_UID_BLAS_OFFSET + 4, task_dm_add);
TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_mul);
TASK(TASK_UID_BLAS_OFFSET + 6, task_dm_conv);
TASK(TASK_UID_BLAS_OFFSET + 7, task_dm_conv_same);
TASK(TASK_UID_BLAS_OFFSET + 8, task_sm_mul);
TASK(TASK_UID_BLAS_OFFSET + 9, task_sm_conv);
TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv_same);

// Private tasks
void task_cleanup_blas();
TASK(TASK_UID_BLAS_OFFSET + 11, task_cleanup_blas);

// Resets a task
static __fram task_t *last_task;
void task_cleanup_blas() {
	// PRINTF("\r\n     Finishing BLAS");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

// Dense scalar addition
void task_ds_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = 0; i < rows; i++) {
		for(uint16_t j = 0; j < cols; j++) {
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense scalar multiplication
void task_ds_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = 0; i < rows; i++) {
		for(uint16_t j = 0; j < cols; j++) {
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_mul(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_MUL(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense scalar division
void task_ds_div() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = 0; i < rows; i++) {
		for(uint16_t j = 0; j < cols; j++) {
			fixed w = F_DIV(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense set mat to all 0s
void task_ds_zero() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = 0; i < rows; i++) {
		for(uint16_t j = 0; j < cols; j++) {
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_st(1);
			MAT_SET(dest, 0, i, j);
		}
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 2);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix addition
void task_dm_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = 0; i < rows; i++) {
		for(uint16_t j = 0; j < cols; j++) {
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, i, j));
			MAT_SET(dest, w, i, j);
		}
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix multiplication
void task_dm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(filter, 0);
	uint16_t cols = MAT_GET_DIM(filter, 1);
	uint16_t dcols = MAT_GET_DIM(dest, 1);
	for(uint16_t i = 0; i < rows; i++) {
		for(uint16_t k = 0; k < dcols; k++) {
			fixed w = 0;
			for(uint16_t j = 0; j < cols; j++) {
				inc_addr_add(2);
				inc_addr_mul(2);
				inc_mul(1);
				inc_add(1);
				inc_ld(3);
				fixed tmp = F_MUL(MAT_GET(filter, i, j), MAT_GET(src, j, k));
				w = F_ADD(w, tmp);
			}
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_st(1);
			MAT_SET(dest, w, i, k);
		}
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution
void task_dm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	for(uint16_t k = 0; k < flayers; k++) {
		for(uint16_t l = 0; l < frows; l++) {
			for(uint16_t n = 0; n < fcols; n++) {
				for(uint16_t i = 0; i < rows; i++) {
					for(uint16_t j = 0; j < cols; j++) {
						fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
						if(k == 0 && l == 0 && n == 0) { // Zero
							MAT_SET(dest, w, i, j);
							continue;
						}
						w = F_ADD(w, MAT_GET(dest, i, j));
						MAT_SET(dest, w, i, j);
					}
				}
			}
		}
	}

	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution, same padding
void task_dm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	for(uint16_t k = 0; k < flayers; k++) {
		for(uint16_t l = 0; l < frows; l++) {
			for(uint16_t n = 0; n < fcols; n++) {
				for(uint16_t i = 0; i < rows; i++) {
					for(uint16_t j = 0; j < cols; j++) {
						fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
						if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
							w = 0;
						}
						if(k == 0 && l == 0 && n == 0) { // Zero
							MAT_SET(dest, w, i, j);
							continue;
						}
						w = F_ADD(w, MAT_GET(dest, i, j));
						MAT_SET(dest, w, i, j);
					}
				}
			}
		}
	}

	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Sparse matrix multiplication
void task_sm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t cols = MAT_GET_DIM(src, 0); // p => j
	uint16_t dcols = MAT_GET_DIM(dest, 1); // p => j
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

	uint16_t pos = 0;
	uint16_t i = 0;
	uint16_t k = 0;
	char zero = 1;

	while(pos < total_elements) {
		inc_addr_add(2);
		inc_addr_mul(1);
		k += filter->sparse.offsets[pos];
		if(k / cols > 0) zero = 1;
		i += k / cols;
		k %= cols;
		// PRINTF("\r\n i: %u k: %u pos: %u val: %i", i, k, pos, MAT_GET(filter, pos));
		for(uint16_t j = 0; j < dcols; j++) {
			inc_addr_add(3);
			inc_addr_mul(2);
			inc_mul(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
			if(!zero) {
				inc_addr_add(1);
				inc_addr_mul(1);
				inc_ld(1);
				inc_add(1);
				w = F_ADD(w, MAT_GET(dest, i, j));
			}
			MAT_SET(dest, w, i, j);
		}
		inc_addr_add(1);
		pos++;
		zero = 0;
	}

	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

	uint16_t idx = 0;
	uint16_t pos = 0;
	char zero = 1;
	while(pos < total_elements) {
		inc_addr_add(1);
		idx += filter->sparse.offsets[pos];
		uint16_t k = idx / (fcols * frows); // Layers
		uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
		inc_addr_mul(2);
		uint16_t n = idx % fcols; // Cols
		// PRINTF("\r\n k: %u l: %u n: %u idx: %u pos: %u val: %i", k, l, n, idx, pos, MAT_GET(filter, pos));
		if(stride[1] + stride[2] > 2) {
			uint16_t i_stride = 0;
			for(uint16_t i = 0; i < rows * stride[1]; i += stride[1]) {
				uint16_t j_stride = 0;
				for(uint16_t j = 0; j < cols * stride[2]; j += stride[2]) {
					inc_addr_add(5);
					inc_addr_mul(3);
					inc_mul(1);
					inc_ld(2);
					inc_st(1);
					if(!zero) {
						inc_addr_mul(1);
						inc_addr_add(1);
						inc_ld(1);
					}
					fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
					w = (zero) ? w : F_ADD(w, MAT_GET(dest, i_stride, j_stride)); // Zero
					MAT_SET(dest, w, i_stride, j_stride);
					j_stride++;
				}
				i_stride++;
			}
		} else {
			for(uint16_t i = 0; i < rows; i++) {
				for(uint16_t j = 0; j < cols; j++) {
					inc_addr_add(5);
					inc_addr_mul(3);
					inc_mul(1);
					inc_ld(2);
					inc_st(1);
					if(!zero) {
						inc_addr_mul(1);
						inc_addr_add(1);
						inc_ld(1);
					}
					fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
					w = (zero) ? w : F_ADD(w, MAT_GET(dest, i, j)); // Zero
					MAT_SET(dest, w, i, j);
				}
			}
		}
		zero = 0;
		inc_addr_add(1);
		pos++;
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

	uint16_t idx = 0;
	uint16_t pos = 0;
	char zero = 1;
	while(pos < total_elements) {
		idx += filter->sparse.offsets[pos];
		uint16_t k = idx / (fcols * frows); // Layers
		uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
		uint16_t n = idx % fcols; // Cols
		inc_addr_mul(2);
		// PRINTF("\r\n k: %u l: %u n: %u idx: %u pos: %u val: %i", k, l, n, idx, pos, MAT_GET(filter, pos));
		if(stride[1] + stride[2] > 2) {
			uint16_t i_stride = 0;
			for(uint16_t i = 0; i < rows * stride[1]; i += stride[1]) {
				uint16_t j_stride = 0;
				for(uint16_t j = 0; j < cols * stride[2]; j += stride[2]) {
					fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
					if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
						w = 0;
					}
					inc_addr_add(5);
					inc_addr_mul(3);
					inc_mul(1);
					inc_ld(2);
					inc_st(1);
					if(!zero) {
						inc_addr_mul(1);
						inc_addr_add(1);
						inc_add(1);
						inc_ld(1);
					}
					w = (zero) ? w : F_ADD(w, MAT_GET(dest, i_stride, j_stride)); // Zero
					MAT_SET(dest, w, i_stride, j_stride);
					j_stride++;
				}
				i_stride++;
			}
		} else {
			for(uint16_t i = 0; i < rows; i++) {
				for(uint16_t j = 0; j < cols; j++) {
					fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
					if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
						w = 0;
					}
					inc_addr_add(5);
					inc_addr_mul(3);
					inc_mul(1);
					inc_ld(2);
					inc_st(1);
					if(!zero) {
						inc_addr_mul(1);
						inc_addr_add(1);
						inc_ld(1);
					}
					w = (zero) ? w : F_ADD(w, MAT_GET(dest, i, j)); // Zero
					MAT_SET(dest, w, i, j);
				}
			}
		}
		zero = 0;
		inc_addr_add(1);
		pos++;
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}