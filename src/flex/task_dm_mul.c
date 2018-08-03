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

TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_mul);

static __fram mat_t buf = {.data = LAYER_BUFFER(3)};
static __fram mat_t *buffer = &buf;

// Dense matrix multiplication
void task_dm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = buffer;
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(filter, 0);
	uint16_t cols = MAT_GET_DIM(filter, 1);
	uint16_t dcols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter, rows, dcols);

	mat_t *tmp = dest;
	if(CUR_SCRATCH[3]) { // Swap buffers
		dest = inter;
		inter = tmp;
	}

	uint16_t k = CUR_SCRATCH[2];
	prof_pulse(0x20);
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		prof_inc("loop_inc", 1, 1);
		for(uint16_t j = CUR_SCRATCH[1]; j < dcols; j = ++CUR_SCRATCH[1]) {
			prof_inc("loop_inc", 1, 1);
			fixed w = F_MUL(MAT_GET(filter, i, k), MAT_GET(src, k, j));
			prof_inc("F_MUL", 1, 1);
			prof_inc("MAT_GET_2D", 2, 2);
			prof_inc("MAT_SET_2D", 1, 1);
			if(k > 0) {
				w = F_ADD(w, MAT_GET(inter, i, j));
				prof_inc("F_ADD", 1, 1);
				prof_inc("MAT_GET_2D", 1, 1);
			}
			MAT_SET(dest, w, i, j);
		}
		CUR_SCRATCH[1] = 0;
	}
	prof_pulse(0x20);

	scratch_bak[0] = 0;
	scratch_bak[2] = k + 1;
	scratch_bak[3] = CUR_SCRATCH[3] ^ 0x01;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	if(k < cols - 1) {
		write_to_gbuf((uint8_t *)(scratch_bak + 3), 
			(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(CUR_SCRATCH[3]) {
		for(uint16_t i = CUR_SCRATCH[4]; i < rows; i = ++CUR_SCRATCH[4]) {
			for(uint16_t j = CUR_SCRATCH[5]; j < dcols; j = ++CUR_SCRATCH[5]) {
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_SCRATCH[5] = 0;
		}
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}