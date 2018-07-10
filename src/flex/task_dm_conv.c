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

TASK(TASK_UID_BLAS_OFFSET + 6, task_dm_conv);

static __fram mat_t buf = {.data = MAT_BUFFER(0)};
static __fram mat_t *inter = &buf;

// Dense matrix convolution
void task_dm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter, rows, cols);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);

	mat_t *tmp = dest;
	if(CUR_SCRATCH[3]) { // Swap buffers
		dest = inter;
		inter = tmp;
	}

	uint16_t k = CUR_SCRATCH[0];
	uint16_t l = CUR_SCRATCH[1];
	uint16_t n = CUR_SCRATCH[2];
	uint16_t i_stride = CUR_SCRATCH[4] / params.stride[1];
	uint16_t j_stride = CUR_SCRATCH[5] / params.stride[2];
	for(uint16_t i = CUR_SCRATCH[4]; 
		i < rows * params.stride[1]; i = (CUR_SCRATCH[4] += params.stride[1])){
		for(uint16_t j = CUR_SCRATCH[5]; 
			j < cols * params.stride[2]; j = (CUR_SCRATCH[5] += params.stride[2])){
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
		j_stride = 0;
		i_stride++;
		CUR_SCRATCH[5] = 0;
	}

	scratch_bak[0] = k;
	scratch_bak[1] = l;
	if(n + 1 == fcols && l + 1 == frows) {
		scratch_bak[0] = k + 1;
		scratch_bak[1] = 0;
	} else if(n + 1 == fcols) {
		scratch_bak[1] = l + 1;
	}
	scratch_bak[2] = (n + 1 == fcols) ? 0 : n + 1;
	scratch_bak[3] = CUR_SCRATCH[3] ^ 0x01;
	scratch_bak[4] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), 
		(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(!(k + 1 == flayers && l + 1 == frows && n + 1 == fcols)) {
		write_to_gbuf((uint8_t *)(scratch_bak + 3), 
			(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(CUR_SCRATCH[3]) {
		for(uint16_t i = CUR_SCRATCH[6]; i < rows; i = (++CUR_SCRATCH[6])){
			for(uint16_t j = CUR_SCRATCH[7]; j < cols; j = (++CUR_SCRATCH[7])){
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_SCRATCH[7] = 0;
		}
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}