#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "blas.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"
#include "profile.h"
#include "cleanup.h"

TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv);
TASK(TASK_UID_BLAS_OFFSET + 11, task_sm_conv_same);

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
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
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
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}