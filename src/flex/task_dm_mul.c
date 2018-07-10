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

TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_mul);

// Dense matrix multiplication
void task_dm_mul() {
	/*mat_t *src = PEEK_STACK(mat_stack, 0);
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
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);*/
}