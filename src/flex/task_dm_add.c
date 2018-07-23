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

TASK(TASK_UID_BLAS_OFFSET + 4, task_dm_add);

// Dense matrix addition
void task_dm_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
		prof_inc("loop_inc", 1, 1);
		for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
			prof_inc("loop_inc", 1, 1);
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, i, j));
			prof_inc("F_ADD", 1, 1);
			prof_inc("F_MUL", 1, 1);
			prof_inc("MAT_GET_2D", 2, 2);
			prof_inc("MAT_SET_2D", 1, 1);
			MAT_SET(dest, w, i, j);
		}
		CUR_SCRATCH[1] = 0;
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}