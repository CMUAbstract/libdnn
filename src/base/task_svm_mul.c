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

	uint16_t cols = MAT_GET_DIM(src, 0);
	uint16_t rows = MAT_GET_DIM(dest, 0);

	for(uint16_t j = 0; j < cols; j++) {
		fixed *dest_ptr = MAT_PTR(dest, 0, 0);
		for(uint16_t i = 0; i < rows; i++) {
			if(j >= (filter->sparse.sizes[i + 1] - filter->sparse.sizes[i])) {
				if(j == 0) *dest_ptr = 0;
				dest_ptr++;
				continue;
			}
			uint16_t col_idx = filter->sparse.sizes[i] + j;
			fixed f = MAT_GET(filter, col_idx);
			fixed w = MAT_GET(src, filter->sparse.offsets[col_idx], 0);
			w = F_MUL(f, w);
			if(j != 0) {
				w = F_ADD(*dest_ptr, w); // Add partial
			}
			*dest_ptr++ = w;
		}
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}