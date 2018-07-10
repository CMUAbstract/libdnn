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

TASK(TASK_UID_BLAS_OFFSET + 9, task_svm_mul);

// Sparse vector-matrix multiplication
void task_svm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0); // n => i

	for(uint16_t i = 0; i < rows; i++) {
		uint16_t start = filter->sparse.sizes[i];
		uint16_t end = filter->sparse.sizes[i + 1];
		fixed w = 0;
		for(uint16_t j = start; j < end; j++) {
			w = F_ADD(w, F_MUL(MAT_GET(filter, j), 
				MAT_GET(src, filter->sparse.offsets[j])));
		}
		MAT_SET(dest, w, i, 0);
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}