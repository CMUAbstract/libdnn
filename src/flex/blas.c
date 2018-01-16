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
static __hifram fixed data2[MAX_MAT_SIZE];
static __fram mat_t m1;
static __fram mat_t m2;
static __fram mat_t *inter1;
static __fram mat_t *inter2;

static __fram uint scratch_bak[SCRATCH_SIZE];

void task_cleanup_blas();
void task_sm_mul_addr();
TASK(TASK_UID_BLAS_OFFSET + 8, task_cleanup_blas);
TASK(TASK_UID_BLAS_OFFSET + 9, task_sm_mul_addr);

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
	inter1 = &m1;
	inter1->data = data1;
	inter2 = &m2;
	inter2->data = data2;
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
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
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
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
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
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
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
			for(uint j = CUR_INFO.scratch[1]; j < dcols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, i, k), MAT_GET(src, k, j));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
			for(uint j = CUR_INFO.scratch[1]; j < dcols; j = ++CUR_INFO.scratch[1]) {
				fixed w = F_MUL(MAT_GET(filter, i, k), MAT_GET(src, k, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k + 1;
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

// Dense Perforated matrix convolution
void task_dm_perf_conv() {
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
				if(i % 2 == 1) {
					MAT_SET(dest, MAT_GET(prev_dest, i - 1, j), i, j);
					continue;
				} else if(j % 2 == 1) {
					MAT_SET(dest, MAT_GET(prev_dest, i, j - 1), i, j);
					continue;
				}
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


// Sparse matrix multiplication
void task_sm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0); // n => i
	uint cols = MAT_GET_DIM(src, 0); // m => k
	uint dcols = MAT_GET_DIM(dest, 1); // p => j
	MAT_RESHAPE(inter1, rows, dcols);

	uint pos = CUR_INFO.scratch[0];
	uint i = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];
	if(pos == 0) {
		char greater = 0;
		while(MAT_GET(filter, pos) == 0) { // Calculate next pos
			greater = 1;
			k += 255;
			pos++;
		}
		if(greater) k--; // Fix bug with idx % 255 == 0
		k += MAT_GET(filter, pos);
		pos++;

		scratch_bak[0] = pos;
		scratch_bak[1] = k / cols;
		scratch_bak[2] = k % cols;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		transition_to(CUR_TASK);
	}

	for(uint j = CUR_INFO.scratch[3]; j < dcols; j = ++CUR_INFO.scratch[3]) {
		fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
		w = F_ADD(w, MAT_GET(dest, i, j));
		MAT_SET(inter1, w, i, j);
		write_to_gbuf((uint8_t *)(inter1->data + dcols * i + j), (uint8_t *)(dest->data + dcols * i + j), sizeof(fixed));
	}
	TRANSITION_TO(task_sm_mul_addr);
}

void task_sm_mul_addr() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint cols = MAT_GET_DIM(src, 0); // m => k
	uint total_elements = MAT_GET_DIM(filter, 0);

	task_t *sm_mul = TASK_REF(task_sm_mul);
	uint orig_pos = sm_mul->info.scratch[0];
	uint pos = orig_pos;
	uint i = sm_mul->info.scratch[1];
	uint k = sm_mul->info.scratch[2];

	char greater = 0;
	while(MAT_GET(filter, pos + 1) == 0) { // Calculate next pos
		greater = 1;
		k += 255;
		pos++;
	}
	if(greater) k--; // Fix bug with idx % 255 == 0
	pos++;
	k += MAT_GET(filter, pos);
	pos++;

	scratch_bak[0] = pos;
	scratch_bak[1] = i + k / cols;
	scratch_bak[2] = k % cols;

	sm_mul->info.scratch[3] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(sm_mul->info.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(sm_mul->info.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(sm_mul->info.scratch + 2), sizeof(uint));
	if(orig_pos < total_elements - 1) TRANSITION_TO(task_sm_mul);
	POP_STACK(mat_stack, 3);
	last_task = sm_mul;
	TRANSITION_TO(task_cleanup_blas);
}

void task_sm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse_dims[1];
	uint fcols = filter->sparse_dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint idx = CUR_INFO.scratch[0]; // WHAT HAPPENS IF THIS OVERFLOWS?
	uint pos = CUR_INFO.scratch[1];
	uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	uint n = idx % fcols; // Cols
	if(pos == 0) {
		char greater = 0;
		while(MAT_GET(filter, pos) == 0) { // Calculate next pos, idx
			greater = 1;
			idx += 255;
			pos++;
		}
		if(greater) idx--; // Fix bug with idx % 255 == 0
		idx += MAT_GET(filter, pos);
		pos++;
		scratch_bak[0] = idx;
		scratch_bak[1] = pos;
		scratch_bak[5] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
		transition_to(CUR_TASK);
	}

	// // PRINTF("\r\n Convolving cols: %u rows: %u idx:  %u pos:  %u k:  %u l:  %u n:  %u tileX:  %u tileY: %u val: %i", 
	// 	rows, cols, idx, pos, k, l, n, 8, 8, MAT_GET(filter, pos));

	mat_t *prev_dest = (CUR_INFO.scratch[4] % 2 == 0) ? inter2 : inter1;
	if(pos < total_elements - 1) {
		dest = (CUR_INFO.scratch[4] % 2 == 0) ? inter1 : inter2; // swapped
	}

	if(CUR_INFO.scratch[5] == 0) {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[3] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[3] = 0;
		}
	}

	char greater = 0;
	while(MAT_GET(filter, pos + 1) == 0) { // Calculate next pos, idx
		greater = 1;
		idx += 255;
		pos++;
	}
	if(greater) idx--; // Fix bug with idx % 255 == 0
	pos++;
	idx += MAT_GET(filter, pos);
	pos++;
	scratch_bak[0] = idx;
	scratch_bak[1] = pos;
	scratch_bak[2] = 0;
	scratch_bak[3] = 0;
	scratch_bak[4] = ~CUR_INFO.scratch[4];
	scratch_bak[5] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(CUR_INFO.scratch[1] < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}