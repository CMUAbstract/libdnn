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
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint16_t idx = CUR_SCRATCH[0];
	uint16_t pos = CUR_SCRATCH[1];
	inc_ld(2);
	char zero = CUR_SCRATCH[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		inc_ld(1);
		inc_addr_add(1);
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(total_elements % 2 == 0 && pos % 2 == 0) { // A
		dest = inter;
		inter = tmp;
	} else if(total_elements % 2 == 1 && pos % 2 == 1) { // B
		dest = inter;
		inter = tmp;
	}

	uint16_t k = idx / (fcols * frows); // Layers
	uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
	uint16_t n = idx % fcols; // Cols
	// PRINTF("\r\nrows: %u cols: %u frows: %u fcols: %u total_elements: %u idx: %u pos: %u k: %u l: %u n: %u val: %i", 
		// rows, cols, frows, fcols, total_elements, idx, pos, k, l, n, MAT_GET(filter, pos));
	if(stride[1] + stride[2] > 2) {
		for(uint16_t i = CUR_SCRATCH[2]; i < rows * stride[1]; i = (CUR_SCRATCH[2] += stride[1])) {
			uint16_t i_stride = i / stride[1];
			for(uint16_t j = CUR_SCRATCH[3]; j < cols * stride[2]; j = (CUR_SCRATCH[3] += stride[2])) {
				uint16_t j_stride = j / stride[2];
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				if(zero == 2) {
					w = F_ADD(w, MAT_GET(inter, i_stride, j_stride));
				}
				MAT_SET(dest, w, i_stride, j_stride);
			}
			CUR_SCRATCH[3] = 0;
		}
	} else {
		for(uint16_t i = CUR_SCRATCH[2]; i < rows; i = ++CUR_SCRATCH[2]) {
			for(uint16_t j = CUR_SCRATCH[3]; j < cols; j = ++CUR_SCRATCH[3]) {
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				if(zero == 2) {
					w = F_ADD(w, MAT_GET(inter, i, j));
				}
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[3] = 0;
		}
	}
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_sm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint16_t idx = CUR_SCRATCH[0];
	uint16_t pos = CUR_SCRATCH[1];
	char zero = CUR_SCRATCH[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	if(total_elements % 2 == 0 && pos % 2 == 0) { // A
		dest = inter;
		inter = tmp;
	} else if(total_elements % 2 == 1 && pos % 2 == 1) { // B
		dest = inter;
		inter = tmp;
	}

	uint16_t k = idx / (fcols * frows); // Layers
	uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
	uint16_t n = idx % fcols; // Cols
	inc_addr_mul(2);
	if(stride[1] + stride[2] > 2) {
		for(uint16_t i = CUR_SCRATCH[2]; i < rows * stride[1]; i = (CUR_SCRATCH[2] += stride[1])) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint16_t j = CUR_SCRATCH[3]; j < cols * stride[2]; j = (CUR_SCRATCH[3] += stride[2])) {
				inc_ld(1);
				inc_st(1);
				inc_add(1);
				uint16_t i_stride = i / stride[1];
				uint16_t j_stride = j / stride[2];
				inc_addr_mul(3);
				inc_addr_add(6);
				inc_addr_mul(4);
				inc_mul(1);
				inc_ld(2);
				inc_st(1);
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
					w = 0;
				}
				if(zero == 2) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_add(1);
					inc_ld(1);
					w = F_ADD(w, MAT_GET(inter, i_stride, j_stride));
				}
				MAT_SET(dest, w, i_stride, j_stride);
			}
			CUR_SCRATCH[3] = 0;
		}
	} else {
		for(uint16_t i = CUR_SCRATCH[2]; i < rows; i = ++CUR_SCRATCH[2]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint16_t j = CUR_SCRATCH[3]; j < cols; j = ++CUR_SCRATCH[3]) {
				inc_ld(1);
				inc_st(1);
				inc_add(1);
				inc_addr_add(6);
				inc_addr_mul(5);
				inc_mul(1);
				inc_ld(2);
				inc_st(1);
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
					w = 0;
				}
				if(zero == 2) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_add(1);
					inc_ld(1);
					w = F_ADD(w, MAT_GET(inter, i, j));
				}
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[3] = 0;
		}
	}
	inc_addr_add(2);
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
	inc_addr_add(3);
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}