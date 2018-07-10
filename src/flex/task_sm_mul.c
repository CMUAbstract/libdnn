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

TASK(TASK_UID_BLAS_OFFSET + 8, task_sm_mul);

// Sparse matrix multiplication
void task_sm_mul() {
	/*mat_t *src = PEEK_STACK(mat_stack, 0);
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
	char zero = CUR_SCRATCH[3];

	if(zero == 0) {
		scratch_bak[2] = filter->sparse.offsets[pos];
		scratch_bak[1] = scratch_bak[2] / cols;
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
		fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
		if(zero == 2) {
			w = F_ADD(w, MAT_GET(inter, i, j));
		}
		MAT_SET(dest, w, i, j);
		write_to_gbuf((uint8_t *)(dest->data + i * dcols + j), (uint8_t *)(inter->data + i * dcols + j), sizeof(uint16_t));
	}

	scratch_bak[0] = pos + 1;
	scratch_bak[2] = k + filter->sparse.offsets[pos + 1];
	scratch_bak[3] = (scratch_bak[2] / cols > 0) ? 1 : 2;
	scratch_bak[1] = i + scratch_bak[2] / cols;
	scratch_bak[2] %= cols;
	scratch_bak[4] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(pos < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);*/
}
