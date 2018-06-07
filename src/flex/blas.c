#include <string.h>
#include <msp430.h>
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

static __fram mat_t m1 = {.data = MAT_BUFFER(0)};
static __fram mat_t m2 = {.data = MAT_BUFFER(1)};
static __fram mat_t *inter1 = &m1;
static __fram mat_t *inter2 = &m2;
static __fram uint scratch_bak[SCRATCH_SIZE];

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

void pulse(uint pin) {
	P6DIR = 0x03;
	P6OUT = pin;
	__delay_cycles(0x100);
	P6OUT = 0x00;
}

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
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
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
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_mul(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_MUL(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
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
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			fixed w = F_DIV(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense set mat to all 0s
void task_ds_zero() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_st(1);
			MAT_SET(dest, 0, i, j);
		}
		CUR_INFO.scratch[1] = 0;
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
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);	
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, i, j));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
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
	uint rows = MAT_GET_DIM(filter, 0);
	uint cols = MAT_GET_DIM(filter, 1);
	uint dcols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter1, rows, dcols);
	MAT_RESHAPE(inter2, rows, dcols);

	uint k = CUR_INFO.scratch[2];
	mat_t *prev_dest = (k % 2 == 0) ? inter2 : inter1;
	if(k < cols - 1) {
		dest = (k % 2 == 0) ? inter1 : inter2;
	}

	if(k > 0) {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint j = CUR_INFO.scratch[1]; j < dcols; j = ++CUR_INFO.scratch[1]) {
				inc_ld(1);
				inc_st(1);
				inc_add(1);
				inc_addr_add(3);
				inc_addr_mul(3);
				inc_mul(1);
				inc_add(1);
				inc_ld(3);
				inc_st(1);
				fixed w = F_MUL(MAT_GET(filter, i, k), MAT_GET(src, k, j));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint j = CUR_INFO.scratch[1]; j < dcols; j = ++CUR_INFO.scratch[1]) {
				inc_ld(1);
				inc_st(1);
				inc_add(1);
				inc_addr_add(2);
				inc_addr_mul(2);
				inc_mul(1);
				inc_ld(2);
				inc_st(1);
				fixed w = F_MUL(MAT_GET(filter, i, k), MAT_GET(src, k, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k + 1;
	inc_addr_add(1);
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	if(k < cols - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution
void task_dm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint k = CUR_INFO.scratch[2];
	uint l = CUR_INFO.scratch[3];
	uint n = CUR_INFO.scratch[4];
	mat_t *prev_dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k;
	scratch_bak[3] = l;
	if(n + 1 == fcols && l + 1 == frows) {
		scratch_bak[3] = 0;
		scratch_bak[2] = k + 1;
	} else if(n + 1 == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + 1 == fcols) ? 0 : n + 1;
	scratch_bak[5] = ~CUR_INFO.scratch[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution
void task_dm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint k = CUR_INFO.scratch[2];
	uint l = CUR_INFO.scratch[3];
	uint n = CUR_INFO.scratch[4];
	mat_t *prev_dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_INFO.scratch[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
					w = 0;
				}
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k;
	scratch_bak[3] = l;
	if(n + 1 == fcols && l + 1 == frows) {
		scratch_bak[3] = 0;
		scratch_bak[2] = k + 1;
	} else if(n + 1 == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + 1 == fcols) ? 0 : n + 1;
	scratch_bak[5] = ~CUR_INFO.scratch[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Sparse matrix multiplication
void task_sm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0); // n => i
	uint cols = MAT_GET_DIM(src, 0); // m => k
	uint dcols = MAT_GET_DIM(dest, 1); // p => j
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, dcols);

	uint pos = CUR_INFO.scratch[0];
	uint i = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];
	inc_ld(3);
	char zero = CUR_INFO.scratch[3];

	if(zero == 0) {
		scratch_bak[2] = filter->sparse.offsets[pos];
		scratch_bak[1] = scratch_bak[2] / cols;
		inc_addr_mul(1);
		inc_ld(1);
		scratch_bak[2] %= cols;
		scratch_bak[3] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
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

	for(uint j = CUR_INFO.scratch[4]; j < dcols; j = ++CUR_INFO.scratch[4]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
		inc_addr_add(3);
		inc_addr_mul(2);
		inc_mul(1);
		inc_ld(2);
		inc_st(1);
		if(zero == 2) {
			inc_add(1);
			inc_addr_mul(1);
			inc_addr_add(1);
			inc_ld(1);
			w = F_ADD(w, MAT_GET(inter, i, j));
		}
		MAT_SET(dest, w, i, j);
		inc_addr_mul(2);
		inc_addr_add(2);
		write_to_gbuf((uint8_t *)(dest->data + i * dcols + j), (uint8_t *)(inter->data + i * dcols + j), sizeof(uint));
	}

	scratch_bak[0] = pos + 1;
	scratch_bak[2] = k + filter->sparse.offsets[pos + 1];
	scratch_bak[3] = (scratch_bak[2] / cols > 0) ? 1 : 2;
	scratch_bak[1] = i + scratch_bak[2] / cols;
	inc_addr_add(3);
	inc_addr_mul(1);
	scratch_bak[2] %= cols;
	scratch_bak[4] = 0;
	inc_addr_add(4);
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	inc_ld(2);
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		inc_ld(1);
		inc_addr_add(1);
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
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

	uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	uint n = idx % fcols; // Cols
	inc_addr_mul(2);
	if(stride[1] + stride[2] > 2) {
		for(uint i = CUR_INFO.scratch[2]; i < rows * stride[1]; i = (CUR_INFO.scratch[2] += stride[1])) {
			uint i_stride = i / stride[1];
			inc_addr_mul(1);
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint j = CUR_INFO.scratch[3]; j < cols * stride[2]; j = (CUR_INFO.scratch[3] += stride[2])) {
				uint j_stride = j / stride[2];
				inc_ld(1);
				inc_st(1);
				inc_add(1);
				inc_addr_mul(2);
				inc_addr_add(6);
				inc_addr_mul(4);
				inc_mul(1);
				inc_ld(2);
				inc_st(1);
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				if(zero == 2) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_add(1);
					inc_ld(1);
					w = F_ADD(w, MAT_GET(inter, i_stride, j_stride));
				}
				MAT_SET(dest, w, i_stride, j_stride);
			}
			CUR_INFO.scratch[3] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
				inc_ld(1);
				inc_st(1);
				inc_add(1);
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				inc_addr_add(6);
				inc_addr_mul(5);
				inc_mul(1);
				inc_ld(2);
				inc_st(1);
				if(zero == 2) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_add(1);
					inc_ld(1);
					w = F_ADD(w, MAT_GET(inter, i, j));
				}
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[3] = 0;
		}
	}
	inc_ld(1);
	inc_addr_add(2);
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
	inc_addr_add(3);
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		inc_ld(1);
		inc_addr_add(1);
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
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

	uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	uint n = idx % fcols; // Cols
	inc_addr_mul(2);
	if(stride[1] + stride[2] > 2) {
		for(uint i = CUR_INFO.scratch[2]; i < rows * stride[1]; i = (CUR_INFO.scratch[2] += stride[1])) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint j = CUR_INFO.scratch[3]; j < cols * stride[2]; j = (CUR_INFO.scratch[3] += stride[2])) {
				inc_ld(1);
				inc_st(1);
				inc_add(1);
				uint i_stride = i / stride[1];
				uint j_stride = j / stride[2];
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
			CUR_INFO.scratch[3] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
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
			CUR_INFO.scratch[3] = 0;
		}
	}
	inc_addr_add(2);
	scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
	scratch_bak[1] = pos + 1;
	scratch_bak[2] = 0;
	scratch_bak[4] = 2;
	inc_addr_add(3);
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}