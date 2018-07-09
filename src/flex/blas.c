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
static __fram uint16_t scratch_bak[SCRATCH_SIZE];

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
TASK(TASK_UID_BLAS_OFFSET + 9, task_svm_mul);
TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv);
TASK(TASK_UID_BLAS_OFFSET + 11, task_sm_conv_same);

// Private tasks
void task_cleanup();
TASK(TASK_UID_BLAS_OFFSET + 11, task_cleanup);

void pulse(uint16_t pin) {
	P6DIR = 0x03;
	P6OUT = pin;
	__delay_cycles(0x100);
	P6OUT = 0x00;
}

// Resets a task
static __fram task_t *last_task;
void task_cleanup() {
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
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
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
		CUR_SCRATCH[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup);
}

// Dense scalar multiplication
void task_ds_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
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
		CUR_SCRATCH[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup);
}

// Dense scalar division
void task_ds_div() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
			fixed w = F_DIV(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_SCRATCH[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup);
}

// Dense set mat to all 0s
void task_ds_zero() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_st(1);
			MAT_SET(dest, 0, i, j);
		}
		CUR_SCRATCH[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 2);
	TRANSITION_TO(task_cleanup);
}

// Dense matrix addition
void task_dm_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		inc_ld(1);
		inc_st(1);
		inc_add(1);
		for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
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
		CUR_SCRATCH[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup);
}

// Dense matrix multiplication
void task_dm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(filter, 0);
	uint16_t cols = MAT_GET_DIM(filter, 1);
	uint16_t dcols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter1, rows, dcols);
	MAT_RESHAPE(inter2, rows, dcols);

	uint16_t k = CUR_SCRATCH[2];
	mat_t *prev_dest = (k % 2 == 0) ? inter2 : inter1;
	if(k < cols - 1) {
		dest = (k % 2 == 0) ? inter1 : inter2;
	}

	if(k > 0) {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint16_t j = CUR_SCRATCH[1]; j < dcols; j = ++CUR_SCRATCH[1]) {
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
			CUR_SCRATCH[1] = 0;
		}
	} else {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint16_t j = CUR_SCRATCH[1]; j < dcols; j = ++CUR_SCRATCH[1]) {
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
			CUR_SCRATCH[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k + 1;
	inc_addr_add(1);
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	if(k < cols - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup);
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
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint16_t k = CUR_SCRATCH[2];
	uint16_t l = CUR_SCRATCH[3];
	uint16_t n = CUR_SCRATCH[4];
	mat_t *prev_dest = (CUR_SCRATCH[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_SCRATCH[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
		}
	} else {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
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
	scratch_bak[5] = ~CUR_SCRATCH[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_SCRATCH + 5), sizeof(uint16_t));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup);
}

// Dense matrix convolution
void task_dm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint16_t k = CUR_SCRATCH[2];
	uint16_t l = CUR_SCRATCH[3];
	uint16_t n = CUR_SCRATCH[4];
	mat_t *prev_dest = (CUR_SCRATCH[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_SCRATCH[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
					w = 0;
				}
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
		}
	} else {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
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
	scratch_bak[5] = ~CUR_SCRATCH[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_SCRATCH + 5), sizeof(uint16_t));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup);
}

// Sparse vector-matrix multiplication
void task_svm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0); // n => i
	uint16_t cols = MAT_GET_DIM(src, 0); // m => j
	MAT_RESHAPE(inter1, rows, 1);

	mat_t *tmp = dest;
	if(CUR_SCRATCH[2]) { // A
		dest = inter;
		inter = tmp;
	}

	uint16_t j = CUR_SCRATCH[1]; // data/col index
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = (++CUR_SCRATCH[0])) {
		if(j >= (filter->sparse.sizes[i + 1] - filter->sparse.sizes[i])) {
			if(j == 0) MAT_SET(dest, F_LIT(0), i, 0); // Empty row
			break;
		}
		uint16_t col_idx = filter->sparse.sizes[i] + j;
		fixed f = MAT_GET(filter, col_idx);
		fixed w = MAT_GET(src, filter->sparse.offsets[col_idx], 0);
		if(j == 0) {
			w = F_MUL(f, w); // Zero array
		} else {
			w = F_ADD(MAT_GET(inter, i, 0), F_MUL(f, w)); // Add partial
		}
		MAT_SET(dest, w, i, 0);
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = j + 1;
	scratch_bak[2] = CUR_SCRATCH[2] ^ 0x01;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	if(j < cols) {
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);
	}
	if(CUR_SCRATCH[2]) {
		for(uint16_t i = CUR_SCRATCH[3]; i < rows; i = ++CUR_SCRATCH[3]) {
			MAT_SET(inter, MAT_GET(dest, i, 0), i, 0);
		}
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup);
}

// Sparse matrix multiplication
void task_sm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0); // n => i
	uint16_t cols = MAT_GET_DIM(src, 0); // m => k
	uint16_t dcols = MAT_GET_DIM(dest, 1); // p => j
	uint16_t total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, dcols);

	uint16_t pos = CUR_SCRATCH[0];
	uint16_t i = CUR_SCRATCH[1];
	uint16_t k = CUR_SCRATCH[2];
	inc_ld(3);
	char zero = CUR_SCRATCH[3];

	if(zero == 0) {
		scratch_bak[2] = filter->sparse.offsets[pos];
		scratch_bak[1] = scratch_bak[2] / cols;
		inc_addr_mul(1);
		inc_ld(1);
		scratch_bak[2] %= cols;
		scratch_bak[3] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
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

	for(uint16_t j = CUR_SCRATCH[4]; j < dcols; j = ++CUR_SCRATCH[4]) {
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
		write_to_gbuf((uint8_t *)(dest->data + i * dcols + j), (uint8_t *)(inter->data + i * dcols + j), sizeof(uint16_t));
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
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup);
}

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
	inc_addr_mul(2);
	if(stride[1] + stride[2] > 2) {
		for(uint16_t i = CUR_SCRATCH[2]; i < rows * stride[1]; i = (CUR_SCRATCH[2] += stride[1])) {
			uint16_t i_stride = i / stride[1];
			inc_addr_mul(1);
			inc_ld(1);
			inc_st(1);
			inc_add(1);
			for(uint16_t j = CUR_SCRATCH[3]; j < cols * stride[2]; j = (CUR_SCRATCH[3] += stride[2])) {
				uint16_t j_stride = j / stride[2];
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
			CUR_SCRATCH[3] = 0;
		}
	}
	inc_ld(1);
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
	last_task = CUR_TASK;
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
		inc_ld(1);
		inc_addr_add(1);
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
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup);
}