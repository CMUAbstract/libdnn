#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "mem.h"
#include "blas.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"
#include "cleanup.h"

TASK(TASK_UID_BLAS_OFFSET + 9, task_svm_mul);

// Sparse vector-matrix multiplication
static __fram fixed val_bak = 0;
static __fram struct {
	uint16_t i;
	uint16_t j;
} pos_bak;

void task_svm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);

	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = (++CUR_SCRATCH[0])) {
		uint16_t start = filter->sparse.sizes[i];
		uint16_t end = filter->sparse.sizes[i + 1];
		uint16_t col_idx = start + CUR_SCRATCH[1];
		fixed *filter_ptr = MAT_PTR(filter, col_idx);
		fixed *dest_ptr = MAT_PTR(dest, i, 0);
		uint16_t *offset = filter->sparse.offsets + col_idx;
		uint16_t j = CUR_SCRATCH[1];
		if(i == pos_bak.i && j == pos_bak.j) { // Restore it i, j are the same
			*dest_ptr = val_bak;
		}
        if(start == end) {
            val_bak = 0;
            pos_bak.i = i;
            *dest_ptr++ = 0;
            continue;
        }
        pos_bak.i = i;
		for(j; j < end - start; j = (++CUR_SCRATCH[1])) {
			fixed w = F_MUL(MAT_GET(src, *offset, 0), *filter_ptr++);
			if(j == 0) {
				val_bak = 0; // Zeroing the vector
			} else {
				val_bak = *dest_ptr;
				w = F_ADD(w, val_bak);
			}
			pos_bak.j = j;
			*dest_ptr = w;
			offset++;
		}
		dest_ptr++;
		CUR_SCRATCH[1] = 0;
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

#if 0
static __fram mat_t buf = {.data = LAYER_BUFFER(3)};
static __fram mat_t *buffer = &buf;

void task_svm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = buffer;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	MAT_RESHAPE(inter, rows, 1);

	mat_t *tmp = dest;
	if(CUR_SCRATCH[2]) { // A
		dest = inter;
		inter = tmp;
	}

	uint16_t j = CUR_SCRATCH[1]; // data/col index
	fixed *inter_ptr = MAT_PTR(inter, CUR_SCRATCH[0], 0);
	fixed *dest_ptr = MAT_PTR(dest, CUR_SCRATCH[0], 0);
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = (++CUR_SCRATCH[0])) {
		if(j >= (filter->sparse.sizes[i + 1] - filter->sparse.sizes[i])) {
			if(j == 0) {
				*dest_ptr++ = 0;
			} else {
				*dest_ptr++ = *inter_ptr++;
			}
			continue;
		}
		uint16_t col_idx = filter->sparse.sizes[i] + j;
		fixed f = MAT_GET(filter, col_idx);
		fixed w = MAT_GET(src, filter->sparse.offsets[col_idx], 0);
		w = F_MUL(f, w);
		if(j != 0) {
			w = F_ADD(*inter_ptr++, w); // Add partial
		}
		*dest_ptr++ = w;
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = j + 1;
	scratch_bak[2] = CUR_SCRATCH[2] ^ 0x01;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	uint16_t cols = MAT_GET_DIM(src, 0);
	if(j < cols) {
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);
	}
	if(CUR_SCRATCH[2]) {
		for(uint16_t i = CUR_SCRATCH[3]; i < rows; i = ++CUR_SCRATCH[3]) {
			MAT_SET(inter, MAT_GET(dest, i, 0), i, 0);
		}
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
#endif
