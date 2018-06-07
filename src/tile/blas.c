#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "blas.h"
#include "mem.h"
#include "buffer.h"
#include "state.h"
#include "misc.h"
#include "profile.h"

static __fram mat_t m1 = {.data = MAT_BUFFER(0)};
static __fram mat_t *inter1 = &m1;
static __fram uint16_t scratch_bak[SCRATCH_SIZE];

// Public tasks
TASK(TASK_UID_BLAS_OFFSET, task_ds_zero);
TASK(TASK_UID_BLAS_OFFSET + 1, task_ds_add);
TASK(TASK_UID_BLAS_OFFSET + 2, task_ds_mul);
TASK(TASK_UID_BLAS_OFFSET + 3, task_ds_div);
TASK(TASK_UID_BLAS_OFFSET + 4, task_dm_add);
TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_mul);
TASK(TASK_UID_BLAS_OFFSET + 6, task_dm_conv);
TASK(TASK_UID_BLAS_OFFSET + 7, task_dm_conv_same); // same padding
TASK(TASK_UID_BLAS_OFFSET + 8, task_sm_mul);
TASK(TASK_UID_BLAS_OFFSET + 9, task_sm_conv);
TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv_same); // same padding

// Private tasks
void task_cleanup_blas();
TASK(TASK_UID_BLAS_OFFSET + 11, task_cleanup_blas);

uint16_t greatest_tile_size(uint16_t a, uint16_t max) {
	uint16_t i = 1;
	uint16_t max_divisor = i;
	while(i < max && i <= a) {
        if(a % i == 0) max_divisor = i;
        i++;
    }
    return max_divisor;
}

uint16_t greatest_common_tile_size(uint16_t a, uint16_t b, uint16_t max) {
	uint16_t i = 1;
	uint16_t max_divisor = i;
	while(i < max && i <= a) {
        if(a % i == 0 && b % i == 0) max_divisor = i;
        i++;
    }
    return max_divisor;
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
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint16_t i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint16_t j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	uint16_t j = CUR_INFO.scratch[1];
	inc_addr_add(1);
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows && CUR_INFO.scratch[1] + tile_size_x ==  cols)) transition_to(CUR_TASK);
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
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint16_t i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint16_t j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_mul(1);
			inc_ld(2);
			inc_st(1);
			fixed w = F_MUL(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	uint16_t j = CUR_INFO.scratch[1];
	inc_addr_add(1);
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows && CUR_INFO.scratch[1] + tile_size_x ==  cols)) transition_to(CUR_TASK);
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
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint16_t i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint16_t j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			fixed w = F_DIV(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	uint16_t j = CUR_INFO.scratch[1];
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows && CUR_INFO.scratch[1] + tile_size_x ==  cols)) transition_to(CUR_TASK);
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
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint16_t i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint16_t j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_st(1);
			MAT_SET(dest, 0, i, j);
		}
	}
	uint16_t j = CUR_INFO.scratch[1];
	inc_addr_add(1);
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows && CUR_INFO.scratch[1] + tile_size_x ==  cols)) transition_to(CUR_TASK);
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
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint16_t i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint16_t j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_add(1);
			inc_ld(2);
			inc_st(1);	
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, i, j));
			MAT_SET(dest, w, i, j);
		}
	}
	uint16_t j = CUR_INFO.scratch[1];
	inc_addr_add(1);
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows && CUR_INFO.scratch[1] + tile_size_x ==  cols)) transition_to(CUR_TASK);
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
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_common_tile_size(rows, dcols, CONFIG_TILE_SIZE);

	MAT_RESHAPE(inter1, tile_size_y, tile_size_x);

	for(uint16_t i = 0; i < tile_size_y; i++) {
		uint16_t idx_i = CUR_INFO.scratch[0] + i;
		inc_addr_add(1);
		for(uint16_t k = 0; k < tile_size_y; k++) {
			uint16_t idx_k = CUR_INFO.scratch[2] + k;
			fixed w = 0;
			for(uint16_t j = 0; j < tile_size_x; j++) {
				inc_addr_add(3);
				inc_addr_mul(2);
				inc_mul(1);
				inc_add(1);
				inc_ld(3);
				uint16_t idx_j = CUR_INFO.scratch[1] + j;
				fixed tmp = F_MUL(MAT_GET(filter, idx_i, idx_j), MAT_GET(src, idx_j, idx_k));
				w = F_ADD(w, tmp);
			}
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_st(2);
			if(CUR_INFO.scratch[1] >= tile_size_x) {
				inc_add(1);
				inc_addr_mul(1);
				inc_addr_add(1);
			}
			w = (CUR_INFO.scratch[1] >= tile_size_x) ? F_ADD(w, MAT_GET(dest, idx_i, idx_k)) : w;
			MAT_SET(inter1, w, i, k);
			inc_addr_mul(2);
			inc_addr_add(3);
			uint16_t where = idx_i * dcols + idx_k;
			write_to_gbuf((uint8_t *)(inter1->data + i * tile_size_x + k), 
				(uint8_t *)(dest->data + where), sizeof(fixed));
		}
	}

	uint16_t i = CUR_INFO.scratch[0];
	uint16_t j = CUR_INFO.scratch[1];
	uint16_t k = CUR_INFO.scratch[2];
	// i
	scratch_bak[0] = CUR_INFO.scratch[0];
	if(CUR_INFO.scratch[1] + tile_size_x == cols) {
		inc_addr_add(1);
		scratch_bak[0] = CUR_INFO.scratch[0] + tile_size_y;
	}
	// j
	scratch_bak[1] = CUR_INFO.scratch[1];
	if(CUR_INFO.scratch[1] + tile_size_x == cols && CUR_INFO.scratch[2] + tile_size_y == dcols) {
		scratch_bak[1] = 0;
	} else if(CUR_INFO.scratch[2] + tile_size_y == dcols) {
		inc_addr_add(1);
		scratch_bak[1] = CUR_INFO.scratch[1] + tile_size_x;
	}
	// k
	inc_addr_add(1);
	scratch_bak[2] = CUR_INFO.scratch[2] + tile_size_y;
	if(CUR_INFO.scratch[2] + tile_size_y == dcols) {
		scratch_bak[2] = 0;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows && 
		CUR_INFO.scratch[1] + tile_size_x == cols && 
		CUR_INFO.scratch[2] + tile_size_y == dcols)) {
		transition_to(CUR_TASK);	
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution
__fram fixed dm_conv_inter = 0;
void task_dm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	uint16_t i = CUR_INFO.scratch[0];
	uint16_t j = CUR_INFO.scratch[1];
	uint16_t tile_size_z = greatest_tile_size(flayers, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(frows, CONFIG_TILE_SIZE);
	uint16_t tile_size_x = greatest_tile_size(fcols, CONFIG_TILE_SIZE);

	fixed w = 0;
	for(uint16_t k = 0; k < tile_size_z; k++) {
		for(uint16_t l = 0; l < tile_size_y; l++) {
			for(uint16_t n = 0; n < tile_size_x; n++) {
				uint16_t idx_k = k + CUR_INFO.scratch[2];
				uint16_t idx_l = l + CUR_INFO.scratch[3];
				uint16_t idx_n = n + CUR_INFO.scratch[4];
				fixed tmp = F_MUL(MAT_GET(src, idx_k, i + idx_l, j + idx_n), MAT_GET(filter, idx_k, idx_l, idx_n));
				w = F_ADD(w, tmp);
			}
		}
	}
	dm_conv_inter = w;
	if(CUR_INFO.scratch[2] >= tile_size_z && // ZERO
		CUR_INFO.scratch[3] >= tile_size_y && 
		CUR_INFO.scratch[4] >= tile_size_y ) {
		dm_conv_inter = MAT_GET(dest, i, j) + w;
	}
	write_to_gbuf((uint8_t *)(&dm_conv_inter), (uint8_t *)(dest->data + cols * i + j), sizeof(fixed));

	uint16_t k = CUR_INFO.scratch[2];
	uint16_t l = CUR_INFO.scratch[3];
	uint16_t n = CUR_INFO.scratch[4];
	scratch_bak[0] = CUR_INFO.scratch[0];
	scratch_bak[1] = CUR_INFO.scratch[1];
	scratch_bak[2] = (l + tile_size_y == frows) ? k + tile_size_z : k;
	if(k + tile_size_z == flayers && l + tile_size_y == frows && n + tile_size_x == fcols) {
		scratch_bak[0] = (j + 1 == cols) ? CUR_INFO.scratch[0] + 1 : CUR_INFO.scratch[0];
		scratch_bak[1] = (j + 1 == cols) ? 0 : CUR_INFO.scratch[1] + 1;
		scratch_bak[2] = 0;
	}
	scratch_bak[3] = l;
	if(n + tile_size_x == fcols && l + tile_size_y == frows) {
		scratch_bak[3] = 0;
	} else if(n + tile_size_x == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + tile_size_x == fcols) ? 0 : n + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + 1 == rows && CUR_INFO.scratch[1] + 1 == cols)) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_dm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	uint16_t i = CUR_INFO.scratch[0];
	uint16_t j = CUR_INFO.scratch[1];
	uint16_t tile_size_z = greatest_tile_size(flayers, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(frows, CONFIG_TILE_SIZE);
	uint16_t tile_size_x = greatest_tile_size(fcols, CONFIG_TILE_SIZE);

	fixed w = 0;
	for(uint16_t k = 0; k < tile_size_z; k++) {
		for(uint16_t l = 0; l < tile_size_y; l++) {
			for(uint16_t n = 0; n < tile_size_x; n++) {
				uint16_t idx_k = k + CUR_INFO.scratch[2];
				uint16_t idx_l = l + CUR_INFO.scratch[3];
				uint16_t idx_n = n + CUR_INFO.scratch[4];
				fixed tmp = F_MUL(MAT_GET(src, idx_k, i + idx_l, j + idx_n), MAT_GET(filter, idx_k, idx_l, idx_n));
				if(i + idx_l >= MAT_GET_DIM(src, 1) || j + idx_n >= MAT_GET_DIM(src, 2)) {
					w = 0;
				}
				w = F_ADD(w, tmp);
			}
		}
	}
	dm_conv_inter = w;
	if(CUR_INFO.scratch[2] >= tile_size_z && // ZERO
		CUR_INFO.scratch[3] >= tile_size_y && 
		CUR_INFO.scratch[4] >= tile_size_y ) {
		dm_conv_inter = MAT_GET(dest, i, j) + w;
	}
	write_to_gbuf((uint8_t *)(&dm_conv_inter), (uint8_t *)(dest->data + cols * i + j), sizeof(fixed));

	uint16_t k = CUR_INFO.scratch[2];
	uint16_t l = CUR_INFO.scratch[3];
	uint16_t n = CUR_INFO.scratch[4];
	scratch_bak[0] = CUR_INFO.scratch[0];
	scratch_bak[1] = CUR_INFO.scratch[1];
	scratch_bak[2] = (l + tile_size_y == frows) ? k + tile_size_z : k;
	if(k + tile_size_z == flayers && l + tile_size_y == frows && n + tile_size_x == fcols) {
		scratch_bak[0] = (j + 1 == cols) ? CUR_INFO.scratch[0] + 1 : CUR_INFO.scratch[0];
		scratch_bak[1] = (j + 1 == cols) ? 0 : CUR_INFO.scratch[1] + 1;
		scratch_bak[2] = 0;
	}
	scratch_bak[3] = l;
	if(n + tile_size_x == fcols && l + tile_size_y == frows) {
		scratch_bak[3] = 0;
	} else if(n + tile_size_x == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + tile_size_x == fcols) ? 0 : n + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint16_t));
	if(!(CUR_INFO.scratch[0] + 1 == rows && CUR_INFO.scratch[1] + 1 == cols)) transition_to(CUR_TASK);
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

	uint16_t rows = MAT_GET_DIM(dest, 0); // n => i
	uint16_t cols = MAT_GET_DIM(src, 0); // m => k
	uint16_t dcols = MAT_GET_DIM(dest, 1); // p => j
	uint16_t total_elements = MAT_GET_DIM(filter, 0);
	uint16_t tile_size_x = greatest_tile_size(dcols, CONFIG_TILE_SIZE);
	MAT_RESHAPE(inter1, rows, dcols);

	uint16_t pos = CUR_INFO.scratch[0];
	uint16_t i = CUR_INFO.scratch[1];
	uint16_t k = CUR_INFO.scratch[2];
	char zero = CUR_INFO.scratch[3];

	if(zero == 0) {
		scratch_bak[2] = filter->sparse.offsets[pos];
		scratch_bak[1] = scratch_bak[2] / cols;
		scratch_bak[2] %= cols;
		scratch_bak[3] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}

	mat_t *tmp = dest;
	// PROBLEM HERE
	if(total_elements % 2 == 0 && pos % 2 == 0) { // A
		dest = inter;
		inter = tmp;
	} else if(total_elements % 2 == 1 && pos % 2 == 1) { // B
		dest = inter;
		inter = tmp;
	}

	for(uint16_t j = CUR_INFO.scratch[4]; j < CUR_INFO.scratch[4] + tile_size_x; j++) {
		inc_addr_add(3);
		inc_addr_mul(2);
		inc_mul(1);
		inc_ld(2);
		inc_st(1);
		fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
		if(zero == 2) {
			inc_addr_add(1);
			inc_addr_mul(1);
			inc_ld(1);
			inc_add(1);	
			w = F_ADD(w, MAT_GET(inter, i, j));
		}
		MAT_SET(dest, w, i, j);
		write_to_gbuf((uint8_t *)(dest->data + i * dcols + j), (uint8_t *)(inter->data + i * dcols + j), sizeof(uint16_t));
	}

	uint16_t j = CUR_INFO.scratch[4];
	scratch_bak[4] = (j + tile_size_x == dcols) ? 0 : j + tile_size_x;
	inc_addr_add(1);
	if(j + tile_size_x == dcols) {
		scratch_bak[0] = pos + 1;
		scratch_bak[2] = k + filter->sparse.offsets[pos + 1];
		scratch_bak[3] = (scratch_bak[2] / cols > 0) ? 1 : 2;
		scratch_bak[1] = i + scratch_bak[2] / cols;
		scratch_bak[2] %= cols;
		inc_addr_add(3);
		inc_addr_mul(1);
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint16_t));
	}
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint16_t));
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

	uint16_t rows = stride[1] * MAT_GET_DIM(dest, 0);
	uint16_t cols = stride[2] * MAT_GET_DIM(dest, 1);
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint16_t idx = CUR_INFO.scratch[0];
	uint16_t pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}

	uint16_t i = CUR_INFO.scratch[2];
	uint16_t j = CUR_INFO.scratch[3];
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
		for(uint16_t ti = 0; ti < tile_size_y; ti++) {
			uint16_t idx_i = i + ti;
			inc_addr_add(1);
			if(idx_i % stride[1] != 0) continue;
			uint16_t strided_i = idx_i / stride[1];
			for(uint16_t tj = 0; tj < tile_size_x; tj++) {
				uint16_t idx_j = j + tj;
				inc_addr_add(1);
				if(idx_j % stride[2] != 0) continue;
				uint16_t strided_j = idx_j / stride[1];
				inc_addr_add(5);
				inc_addr_mul(3);
				inc_mul(1);
				inc_ld(2);
				inc_st(1);	
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, idx_i + l, idx_j + n));
				if(zero == 2) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_ld(1);
					inc_add(1);
					w = F_ADD(w, MAT_GET(inter, strided_i, strided_j));
				}
				MAT_SET(dest, w, strided_i, strided_j);
				inc_addr_add(2);
				inc_addr_mul(2);
				inc_st(1);
				write_to_gbuf((uint8_t *)(dest->data + strided_i * cols + strided_j), 
					(uint8_t *)(inter->data + strided_i * cols + strided_j), sizeof(uint16_t));
			}
		}
	} else {
		for(uint16_t ti = 0; ti < tile_size_y; ti++) {
			inc_addr_add(1);
			for(uint16_t tj = 0; tj < tile_size_x; tj++) {
				uint16_t idx_i = i + ti;
				inc_addr_add(1);
				uint16_t idx_j = j + tj;
				inc_addr_add(5);
				inc_addr_mul(3);
				inc_mul(1);
				inc_ld(2);
				inc_st(1);	
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, idx_i + l, idx_j + n));
				if(zero == 2) {
					inc_addr_mul(1);
					inc_addr_add(1);
					inc_add(1);
					inc_ld(1);
					w = F_ADD(w, MAT_GET(inter, idx_i, idx_j));
				}
				MAT_SET(dest, w, idx_i, idx_j);
				inc_addr_add(2);
				inc_addr_mul(2);
				inc_st(1);
				write_to_gbuf((uint8_t *)(dest->data + idx_i * cols + idx_j), (uint8_t *)(inter->data + idx_i * cols + idx_j), sizeof(uint16_t));
			}
		}
	}

	scratch_bak[2] = (j + tile_size_x == cols) ? i + tile_size_y : i;
	scratch_bak[3] = (j + tile_size_x == cols) ? 0 : j + tile_size_x;
	inc_addr_add(1);

	if(j + tile_size_x == cols && i + tile_size_y == rows) {
		inc_addr_add(2);
		scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
		scratch_bak[1] = pos + 1;
		scratch_bak[2] = 0;
		scratch_bak[4] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint16_t));
	}
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint16_t));
	if(j + tile_size_x != cols || i + tile_size_y != rows || pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv_same() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = inter1;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint16_t idx = CUR_INFO.scratch[0];
	uint16_t pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}

	uint16_t i = CUR_INFO.scratch[2];
	uint16_t j = CUR_INFO.scratch[3];
	mat_t *tmp = dest;
	// Problem here
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
	for(uint16_t ti = 0; ti < tile_size_y; ti++) {
		inc_addr_add(1);
		for(uint16_t tj = 0; tj < tile_size_x; tj++) {
			uint16_t idx_i = i + ti;
			inc_addr_add(1);
			uint16_t idx_j = j + tj;
			inc_addr_add(5);
			inc_addr_mul(3);
			inc_mul(1);
			inc_ld(2);
			inc_st(1);	
			fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, idx_i + l, idx_j + n));
			if(idx_i + l >= MAT_GET_DIM(src, 1) || idx_j + n >= MAT_GET_DIM(src, 2)) {
				w = 0;
			}
			if(zero == 2) {
				inc_addr_mul(1);
				inc_addr_add(1);
				inc_add(1);
				inc_ld(1);
				w = F_ADD(w, MAT_GET(inter, idx_i, idx_j));
			}
			MAT_SET(dest, w, idx_i, idx_j);
			inc_addr_add(2);
			inc_addr_mul(2);
			inc_st(1);
			write_to_gbuf((uint8_t *)(dest->data + idx_i * cols + idx_j), (uint8_t *)(inter->data + idx_i * cols + idx_j), sizeof(uint16_t));
		}
	}

	scratch_bak[2] = (j + tile_size_x == cols) ? i + tile_size_y : i;
	scratch_bak[3] = (j + tile_size_x == cols) ? 0 : j + tile_size_x;
	inc_addr_add(1);

	if(j + tile_size_x == cols && i + tile_size_y == rows) {
		inc_addr_add(2);
		scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
		scratch_bak[1] = pos + 1;
		scratch_bak[2] = 0;
		scratch_bak[4] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint16_t));
	}
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint16_t));
	if(j + tile_size_x != cols || i + tile_size_y != rows || pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}