#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>

#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "fixed.h"
#include "mat.h"

static __hifram fixed data1[MAX_MAT_SIZE];
static __fram mat_t m;
static __fram mat_t *inter1;

static __fram uint scratch_bak[SCRATCH_SIZE];

void task_cleanup_blas();
TASK(TASK_UID_BLAS_OFFSET + 8, task_cleanup_blas);

uint greatest_tile_size(uint a, uint max) {
	uint i = 1;
	uint max_divisor = i;
	while(i < max && i <= a) {
        if(a % i == 0) max_divisor = i;
        i++;
    }
    return max_divisor;
}

uint greatest_common_tile_size(uint a, uint b, uint max) {
	uint i = 1;
	uint max_divisor = i;
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

// Initialize the blas library
void task_init_blas() {
	PRINTF("\r\n Initializing BLAS");
	inter1 = &m;
	inter1->data = data1;
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense scalar addition
void task_ds_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	uint tile_size_x = greatest_tile_size(cols, MAX_TILE_SIZE);
	uint tile_size_y = greatest_tile_size(rows, MAX_TILE_SIZE);
	for(uint i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	uint j = CUR_INFO.scratch[1];
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	if(!(CUR_INFO.scratch[0] + tile_size_y == rows && CUR_INFO.scratch[1] + tile_size_x ==  cols)) transition_to(CUR_TASK);
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
	uint tile_size_x = greatest_tile_size(cols, MAX_TILE_SIZE);
	uint tile_size_y = greatest_tile_size(rows, MAX_TILE_SIZE);
	for(uint i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			MAT_SET(dest, 0, i, j);
		}
	}
	uint j = CUR_INFO.scratch[1];
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
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
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	uint tile_size_x = greatest_tile_size(cols, MAX_TILE_SIZE);
	uint tile_size_y = greatest_tile_size(rows, MAX_TILE_SIZE);
	for(uint i = CUR_INFO.scratch[0]; i < CUR_INFO.scratch[0] + tile_size_y; i++) {
		for(uint j = CUR_INFO.scratch[1]; j < CUR_INFO.scratch[1] + tile_size_x; j++) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, i, j));
			MAT_SET(dest, w, i, j);
		}
	}
	uint j = CUR_INFO.scratch[1];
	scratch_bak[0] = (j + tile_size_x == cols) ? CUR_INFO.scratch[0] + tile_size_y : CUR_INFO.scratch[0];
	scratch_bak[1] = (j + tile_size_x == cols) ? 0 : CUR_INFO.scratch[1] + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
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
	uint rows = MAT_GET_DIM(filter, 0);
	uint cols = MAT_GET_DIM(filter, 1);
	uint dcols = MAT_GET_DIM(dest, 1);
	uint tile_size_x = greatest_tile_size(cols, MAX_TILE_SIZE);
	uint tile_size_y = greatest_common_tile_size(rows, dcols, MAX_TILE_SIZE);

	MAT_RESHAPE(inter1, tile_size_y, tile_size_x);

	for(uint i = 0; i < tile_size_y; i++) {
		uint idx_i = CUR_INFO.scratch[0] + i;
		for(uint k = 0; k < tile_size_y; k++) {
			uint idx_k = CUR_INFO.scratch[2] + k;
			fixed w = 0;
			for(uint j = 0; j < tile_size_x; j++) {
				uint idx_j = CUR_INFO.scratch[1] + j;
				fixed tmp = F_MUL(MAT_GET(filter, idx_i, idx_j), MAT_GET(src, idx_j, idx_k));
				w = F_ADD(w, tmp);
			}
			w = (CUR_INFO.scratch[1] >= tile_size_x) ? F_ADD(w, MAT_GET(dest, idx_i, idx_k)) : w;
			MAT_SET(inter1, w, i, k);
			uint where = idx_i * dcols + idx_k;
			write_to_gbuf((uint8_t *)(inter1->data + i * tile_size_x + k), 
				(uint8_t *)(dest->data + where), sizeof(fixed));
		}
	}

	uint i = CUR_INFO.scratch[0];
	uint j = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];
	// i
	scratch_bak[0] = CUR_INFO.scratch[0];
	if(CUR_INFO.scratch[1] + tile_size_x == cols) {
		scratch_bak[0] = CUR_INFO.scratch[0] + tile_size_y;
	}
	// j
	scratch_bak[1] = CUR_INFO.scratch[1];
	if(CUR_INFO.scratch[1] + tile_size_x == cols && CUR_INFO.scratch[2] + tile_size_y == dcols) {
		scratch_bak[1] = 0;
	} else if(CUR_INFO.scratch[2] + tile_size_y == dcols) {
		scratch_bak[1] = CUR_INFO.scratch[1] + tile_size_x;
	}
	// k
	scratch_bak[2] = CUR_INFO.scratch[2] + tile_size_y;
	if(CUR_INFO.scratch[2] + tile_size_y == dcols) {
		scratch_bak[2] = 0;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
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

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	uint i = CUR_INFO.scratch[0];
	uint j = CUR_INFO.scratch[1];
	uint tile_size_z = greatest_tile_size(flayers, MAX_TILE_SIZE);
	uint tile_size_y = greatest_tile_size(frows, MAX_TILE_SIZE);
	uint tile_size_x = greatest_tile_size(fcols, MAX_TILE_SIZE);

	fixed w = 0;
	for(uint k = 0; k < tile_size_z; k++) {
		for(uint l = 0; l < tile_size_y; l++) {
			for(uint n = 0; n < tile_size_x; n++) {
				uint idx_k = k + CUR_INFO.scratch[2];
				uint idx_l = l + CUR_INFO.scratch[3];
				uint idx_n = n + CUR_INFO.scratch[4];
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

	uint k = CUR_INFO.scratch[2];
	uint l = CUR_INFO.scratch[3];
	uint n = CUR_INFO.scratch[4];
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
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
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

	uint rows = MAT_GET_DIM(dest, 0); // n => i
	uint cols = MAT_GET_DIM(src, 0); // m => k
	uint dcols = MAT_GET_DIM(dest, 1); // p => j
	uint total_elements = MAT_GET_DIM(filter, 0);
	uint tile_size_x = greatest_tile_size(dcols, MAX_TILE_SIZE);
	MAT_RESHAPE(inter1, rows, dcols);

	uint pos = CUR_INFO.scratch[0];
	uint i = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];
	char zero = CUR_INFO.scratch[3];

	if(zero == 0) {
		scratch_bak[2] = filter->sparse.offsets[pos];
		scratch_bak[1] = scratch_bak[2] / cols;
		scratch_bak[2] %= cols;
		scratch_bak[3] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
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

	for(uint j = CUR_INFO.scratch[4]; j < CUR_INFO.scratch[4] + tile_size_x; j++) {
		fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
		if(zero == 2) {
			w = F_ADD(w, MAT_GET(inter, i, j));
		} else {
			// Pretty sweet double buffering trick here, lazy update
			MAT_SET(dest, MAT_GET(inter, i - 1, j), i - 1, j);
		}
		MAT_SET(dest, w, i, j);
	}

	uint j = CUR_INFO.scratch[4];
	scratch_bak[4] = (j + tile_size_x == dcols) ? 0 : j + tile_size_x;
	if(j + tile_size_x == dcols) {
		scratch_bak[0] = pos + 1;
		scratch_bak[2] = k + filter->sparse.offsets[pos + 1];
		scratch_bak[3] = (scratch_bak[2] / cols > 0) ? 1 : 2;
		scratch_bak[1] = i + scratch_bak[2] / cols;
		scratch_bak[2] %= cols;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	}
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
	uint tile_size_x = greatest_tile_size(cols, MAX_TILE_SIZE);
	uint tile_size_y = greatest_tile_size(rows, MAX_TILE_SIZE);

	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);

	uint idx = CUR_INFO.scratch[0];
	uint pos = CUR_INFO.scratch[1];
	char zero = CUR_INFO.scratch[4];
	if(zero == 0) {
		scratch_bak[0] = filter->sparse.offsets[pos];
		scratch_bak[4] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		transition_to(CUR_TASK);
	}

	uint i = CUR_INFO.scratch[2];
	uint j = CUR_INFO.scratch[3];
	mat_t *tmp = dest;
	// Problem here
	if(total_elements % 2 == 0 && pos % 2 == 0) { // A
		PRINTF("\r\n SWAPPED");
		dest = inter;
		inter = tmp;
	} else if(total_elements % 2 == 1 && pos % 2 == 1) { // B
		dest = inter;
		inter = tmp;
	} else {
		PRINTF("\r\n NO");
	}

	uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	uint n = idx % fcols; // Cols
	for(uint ti = 0; ti < tile_size_y; ti++) {
		for(uint tj = 0; tj < tile_size_x; tj++) {
			uint idx_i = i + ti;
			uint idx_j = j + tj;
			fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, idx_i + l, idx_j + n));
			if(zero == 2) {
				w = F_ADD(w, MAT_GET(inter, idx_i, idx_j));
				int prev_idx_j = j - tile_size_x + tj;
				if(j == 0) {
					uint prev_idx_i = rows - tile_size_y + ti;
					prev_idx_j = cols - tile_size_x + tj;
					MAT_SET(dest, MAT_GET(inter, prev_idx_i, prev_idx_j), prev_idx_i, prev_idx_j);
				} else {
					MAT_SET(inter, MAT_GET(dest, idx_i, prev_idx_j), idx_i, prev_idx_j);
				}
			}
			MAT_SET(dest, w, idx_i, idx_j);
		}
	}

	scratch_bak[2] = (j + tile_size_x == cols) ? i + tile_size_y : i;
	scratch_bak[3] = (j + tile_size_x == cols) ? 0 : j + tile_size_x;

	if(j + tile_size_x == cols && i + tile_size_y == rows) {
		scratch_bak[0] = idx + filter->sparse.offsets[pos + 1];
		scratch_bak[1] = pos + 1;
		scratch_bak[2] = 0;
		scratch_bak[4] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	}
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	if(j + tile_size_x != cols || i + tile_size_y != rows || pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}