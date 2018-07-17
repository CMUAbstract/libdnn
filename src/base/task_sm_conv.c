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

TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv);

// Sparse Matrix Multiplication
void task_sm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

	uint16_t idx = 0;
	uint16_t pos = 0;
	char zero = 1;
	while(pos < total_elements) {
		idx += filter->sparse.offsets[pos];
		uint16_t k = idx / (fcols * frows); // Layers
		uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
		uint16_t n = idx % fcols; // Cols
		// PRINTF("\r\n k: %u l: %u n: %u idx: %u pos: %u val: %i", 
		// 	k, l, n, idx, pos, MAT_GET(filter, pos));
		fixed f = MAT_GET(filter, pos);
		fixed *dest_ptr = MAT_PTR(dest, 0, 0);
		for(uint16_t i = 0; i < rows * params.stride[1]; i += params.stride[1]) {
			fixed *src_ptr = MAT_PTR(src, k, i + l, n);
			for(uint16_t j = 0; j < cols * params.stride[2]; j += params.stride[2]) {
				fixed w = 0;
				if(!params.same_padding || (i + l < MAT_GET_DIM(src, 1) && 
					j + n < MAT_GET_DIM(src, 2))) {
					w = F_MUL(f, *src_ptr);
				}
				if(!zero) {
					w = F_ADD(w, *dest_ptr); // Zero
				}
				*dest_ptr = w;
				dest_ptr++;
				src_ptr += params.stride[2];
			}
		}
		zero = 0;
		pos++;
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}