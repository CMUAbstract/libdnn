#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "blas.h"
#include "mem.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"
#include "profile.h"
#include "cleanup.h"

TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv);

static __fram mat_t buf = {.data = MAT_BUFFER(0)};
static __fram mat_t *buffer = &buf;

void task_sm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *inter = buffer;
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);
	MAT_RESHAPE(inter, rows, cols);

	uint16_t frows = filter->sparse.dims[1];
	uint16_t fcols = filter->sparse.dims[2];
	uint16_t total_elements = MAT_GET_DIM(filter, 0);

	mat_t *tmp = dest;
	if(CUR_SCRATCH[2]) { // Swap buffers
		dest = inter;
		inter = tmp;
	}

	uint16_t pos = CUR_SCRATCH[0];
	uint16_t idx = CUR_SCRATCH[1];
	bool zero = false;
	if(pos == 0) {
		zero = true;
		idx += filter->sparse.offsets[pos];
	}
	uint16_t k = idx / (fcols * frows); // Layers
	uint16_t l = (idx % (fcols * frows)) / fcols; // Rows
	uint16_t n = idx % fcols; // Cols

	uint16_t i_stride = CUR_SCRATCH[3] / params.stride[1];
	uint16_t j_stride = CUR_SCRATCH[4] / params.stride[2];
	fixed f = MAT_GET(filter, pos);
	fixed *inter_ptr = MAT_PTR(inter, i_stride, j_stride);
	fixed *dest_ptr = MAT_PTR(dest, i_stride, j_stride);
	for(uint16_t i = CUR_SCRATCH[3]; 
		i < rows * params.stride[1]; i = (CUR_SCRATCH[3] += params.stride[1])) {
		prof_inc("loop_add", 1, 1);	
		fixed *src_ptr = MAT_PTR(src, k, i + l, CUR_SCRATCH[4] + n);
		prof_inc("MAT_GET_3D", 1, 1);	
		prof_inc("add", 2, 2);
		prof_inc("ld", 1, 1);
		for(uint16_t j = CUR_SCRATCH[4]; 
			j < cols * params.stride[2]; j = (CUR_SCRATCH[4] += params.stride[2])) {
			prof_inc("loop_add", 1, 1);	
			fixed w = 0;
			prof_inc("ld", 2, 2);
			prof_inc("add", 2, 2);
			if(!params.same_padding || (i + l < MAT_GET_DIM(src, 1) && 
				j + n < MAT_GET_DIM(src, 2))) {
				w = F_MUL(f, *src_ptr);
				prof_inc("ld", 1, 1);
				prof_inc("F_MUL", 1, 1);
			}
			if(!zero) {
				w = F_ADD(w, *inter_ptr); // Zero
				prof_inc("ld", 1, 1);
				prof_inc("F_ADD", 1, 1);
				inter_ptr++;
			}
			*dest_ptr = w;
			prof_inc("st", 1, 1);
			dest_ptr++;
			prof_inc("inc", 1, 1);
			src_ptr += params.stride[2];
			prof_inc("ld", 1, 1);
			prof_inc("add", 1, 1);
		}
		CUR_SCRATCH[4] = 0;
	}

	scratch_bak[0] = pos + 1;
	scratch_bak[1] = idx + filter->sparse.offsets[pos + 1];

	scratch_bak[2] = CUR_SCRATCH[2] ^ 0x01;
	scratch_bak[3] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), 
		(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	if(pos < total_elements - 1) {
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(CUR_SCRATCH[2]) {
		for(uint16_t i = CUR_SCRATCH[5]; i < rows; i = (++CUR_SCRATCH[5])){
			for(uint16_t j = CUR_SCRATCH[6]; j < cols; j = (++CUR_SCRATCH[6])){
				MAT_SET(inter, MAT_GET(dest, i, j), i, j);
			}
			CUR_SCRATCH[6] = 0;
		}
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
