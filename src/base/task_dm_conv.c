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

TASK(TASK_UID_BLAS_OFFSET + 6, task_dm_conv);

// Dense matrix convolution
void task_dm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	for(uint16_t k = 0; k < flayers; k++) {
		for(uint16_t l = 0; l < frows; l++) {
			for(uint16_t n = 0; n < fcols; n++) {
				uint16_t i_stride = 0;
				for(uint16_t i = 0; 
					i < rows * params.stride[1]; i += params.stride[1]) {
					uint16_t j_stride = 0;
					for(uint16_t j = 0;
						j < cols * params.stride[2]; j += params.stride[2]) {
						fixed w = 0;
						if(!params.same_padding || (i + l < MAT_GET_DIM(src, 1) && 
							j + n < MAT_GET_DIM(src, 2))) {
							w = F_MUL(MAT_GET(filter, k, l, n), 
								MAT_GET(src, k, i + l, j + n));
						}
						if(k == 0 && l == 0 && n == 0) { // Zero
							MAT_SET(dest, w, i_stride, j_stride);
							j_stride++;
							continue;
						}
						w = F_ADD(w, MAT_GET(dest, i_stride, j_stride));
						MAT_SET(dest, w, i_stride, j_stride);
						j_stride++;
					}
					i_stride++;
				}
			}
		}
	}

	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
