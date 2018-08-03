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
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(filter, 0);
	uint16_t cols = MAT_GET_DIM(filter, 1);
	uint16_t dcols = MAT_GET_DIM(dest, 1);
	prof_pulse(0x20);
	for(uint16_t i = 0; i < rows; i++) {
		for(uint16_t k = 0; k < dcols; k++) {
			fixed w = 0;
			for(uint16_t j = 0; j < cols; j++) {
				fixed tmp = F_MUL(MAT_GET(filter, i, j), MAT_GET(src, j, k));
				w = F_ADD(w, tmp);
			}
			MAT_SET(dest, w, i, k);
		}
	}
	prof_pulse(0x20);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}