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
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t cols = MAT_GET_DIM(src, 0); // p => j
	uint16_t dcols = MAT_GET_DIM(dest, 1); // p => j
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

	uint16_t pos = 0;
	uint16_t i = 0;
	uint16_t k = 0;
	char zero = 1;

	while(pos < total_elements) {
		k += filter->sparse.offsets[pos];
		if(k / cols > 0) zero = 1;
		i += k / cols;
		k %= cols;
		// PRINTF("\r\n i: %u k: %u pos: %u val: %i", i, k, pos, MAT_GET(filter, pos));
		for(uint16_t j = 0; j < dcols; j++) {
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
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}