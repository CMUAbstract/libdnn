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
TASK(TASK_UID_BLAS_OFFSET + 8, task_cleanup_blas);

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
	for(uint i = 0; i < rows; i++) {
		for(uint j = 0; j < cols; j++) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
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
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = 0; i < rows; i++) {
		for(uint j = 0; j < cols; j++) {
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
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = 0; i < rows; i++) {
		for(uint j = 0; j < cols; j++) {
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
	uint rows = MAT_GET_DIM(filter, 0);
	uint cols = MAT_GET_DIM(filter, 1);
	uint dcols = MAT_GET_DIM(dest, 1);
	for(uint i = 0; i < rows; i++) {
		for(uint k = 0; k < dcols; k++) {
			fixed w = 0;
			for(uint j = 0; j < cols; j++) {
				fixed tmp = F_MUL(MAT_GET(filter, i, j), MAT_GET(src, j, k));
				w = F_ADD(w, tmp);
			}
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

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	for(uint k = 0; k < flayers; k++) {
		for(uint l = 0; l < frows; l++) {
			for(uint n = 0; n < fcols; n++) {
				for(uint i = 0; i < rows; i++) {
					for(uint j = 0; j < cols; j++) {
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


void task_dm_conv1d() {
#if 0
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint total_elements = MAT_GET_DIM(filter, 0);
	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	
	uint k = 0;
	uint l = 0;
	uint n = 0;
	char zero = 1;
	for(uint pos = 0; pos < total_elements; pos++) {
		fixed f = MAT_GET(filter, pos);
		for(uint i = 0; i < rows; i++) {
			for(uint j = 0; j < cols; j++) {
				fixed w = F_MUL(f, MAT_GET(src, k, i + l, j + n));
				w = (zero) ? w : F_ADD(w, MAT_GET(dest, i, j)); // Zero
				MAT_SET(dest, w, i, j);
			}
		}
		zero = 0;
		k += stride[0];
		l += stride[1];
		n += stride[2];
	}

	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
#endif
}

// Sparse matrix multiplication
void task_sm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint cols = MAT_GET_DIM(src, 0); // p => j
	uint dcols = MAT_GET_DIM(dest, 1); // p => j
	uint total_elements = MAT_GET_DIM(filter, 0);

	uint pos = 0;
	uint i = 0;
	uint k = 0;
	char zero = 1;

	while(pos < total_elements) {
		k += filter->sparse.offsets[pos];
		if(k / cols > 0) zero = 1;
		i += k / cols;
		k %= cols;
		PRINTF("\r\n i: %u k: %u pos: %u val: %i", i, k, pos, MAT_GET(filter, pos));
		for(uint j = 0; j < dcols; j++) {
			fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
			if(!zero) {
				w = F_ADD(w, MAT_GET(dest, i, j));
			}
			MAT_SET(dest, w, i, j);
		}
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

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);
	uint frows = filter->sparse.dims[1];
	uint fcols = filter->sparse.dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);

	uint idx = 0;
	uint pos = 0;
	char zero = 1;
	while(pos < total_elements) {
		idx += filter->sparse.offsets[pos];
		uint k = idx / (fcols * frows); // Layers
		uint l = (idx % (fcols * frows)) / fcols; // Rows
		uint n = idx % fcols; // Cols
		PRINTF("\r\n k: %u l: %u n: %u idx: %u pos: %u val: %i", k, l, n, idx, pos, MAT_GET(filter, pos));
		for(uint i = 0; i < rows; i++) {
			for(uint j = 0; j < cols; j++) {
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				w = (zero) ? w : F_ADD(w, MAT_GET(dest, i, j)); // Zero
				MAT_SET(dest, w, i, j);
			}
		}
		zero = 0;
		pos++;
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}