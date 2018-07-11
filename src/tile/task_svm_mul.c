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
#include "profile.h"
#include "cleanup.h"
#include "tile.h"

TASK(TASK_UID_BLAS_OFFSET + 9, task_svm_mul);

static __fram mat_t buf = {.data = LAYER_BUFFER(3)};
static __fram mat_t *inter = &buf;

// Sparse vector-matrix multiplication
void task_svm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0); // n => i
	uint16_t cols = MAT_GET_DIM(src, 0); // m => j
	uint16_t tile_size = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	MAT_RESHAPE(inter, rows, 1);

	uint16_t cur_row = CUR_SCRATCH[0]; // row
	uint16_t j = CUR_SCRATCH[1]; // data/col index
	for(uint16_t i = cur_row; i < cur_row + tile_size; i++) {
		if(j >= (filter->sparse.sizes[i + 1] - filter->sparse.sizes[i])) {
			if(j == 0) MAT_SET(dest, F_LIT(0), i, 0); // Empty row
			else MAT_SET(dest, MAT_GET(inter, i, 0), i, 0);
			continue;
		}
		uint16_t col_idx = filter->sparse.sizes[i] + j;
		fixed f = MAT_GET(filter, col_idx);
		fixed w = MAT_GET(src, filter->sparse.offsets[col_idx], 0);
		w = F_MUL(f, w);
		if(j != 0) {
			w = F_ADD(MAT_GET(inter, i, 0), w); // Add partial
		}
		MAT_SET(inter, w, i, 0);
		write_to_gbuf((uint8_t *)(inter->data + i), 
			(uint8_t *)(dest->data + i), sizeof(fixed));
	}

	scratch_bak[0] = cur_row + tile_size;
	if(cur_row + tile_size == rows) {
		scratch_bak[0] = 0;
		scratch_bak[1] = j + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	}
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	if(j < cols - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}