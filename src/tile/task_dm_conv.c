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
#include "tile.h"

TASK(TASK_UID_BLAS_OFFSET + 6, task_dm_conv);

static __fram fixed inter;
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

	uint16_t i = CUR_SCRATCH[0];
	uint16_t j = CUR_SCRATCH[1];
	uint16_t tile_size_z = greatest_tile_size(flayers, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(frows, CONFIG_TILE_SIZE);
	uint16_t tile_size_x = greatest_tile_size(fcols, CONFIG_TILE_SIZE);

	inter = 0;
	for(uint16_t k = CUR_SCRATCH[2]; k < CUR_SCRATCH[2] + tile_size_z; k++) {
		for(uint16_t l = CUR_SCRATCH[3]; l < CUR_SCRATCH[3] + tile_size_y; l++) {
			for(uint16_t n = CUR_SCRATCH[4]; n < CUR_SCRATCH[4] + tile_size_x; n++) {
				if(!params.same_padding || (i + l < MAT_GET_DIM(src, 1) && 
				j + n < MAT_GET_DIM(src, 2))) {
					inter = F_ADD(inter, F_MUL(MAT_GET(filter, k, l, n), 
						MAT_GET(src, k, i + l, j + n)));
				}
			}
		}
	}
	uint16_t i_stride = i / params.stride[1];
	uint16_t j_stride = j / params.stride[2];
	if(CUR_SCRATCH[2] != 0 && CUR_SCRATCH[3] != 0 && CUR_SCRATCH[4] != 0) {
		inter = F_ADD(inter, MAT_GET(dest, i_stride, j_stride));
	}
	write_to_gbuf((uint8_t *)(&inter), 
		(uint8_t *)(dest->data + cols * i_stride + j_stride), sizeof(fixed));

	uint16_t k = CUR_SCRATCH[2];
	uint16_t l = CUR_SCRATCH[3];
	uint16_t n = CUR_SCRATCH[4];
	scratch_bak[0] = CUR_SCRATCH[0];
	scratch_bak[1] = CUR_SCRATCH[1];
	scratch_bak[2] = (l + tile_size_y == frows) ? k + tile_size_z : k;
	if(k + tile_size_z == flayers && l + tile_size_y == frows 
		&& n + tile_size_x == fcols) {
		scratch_bak[0] = CUR_SCRATCH[0];
		scratch_bak[1] = CUR_SCRATCH[1] + params.stride[2];
		if(j + params.stride[2] >= params.stride[2] * cols) {
			scratch_bak[0] = CUR_SCRATCH[0] + params.stride[1];
			scratch_bak[1] = 0;
		}
		scratch_bak[2] = 0;
	}
	scratch_bak[3] = l;
	if(n + tile_size_x == fcols && l + tile_size_y == frows) {
		scratch_bak[3] = 0;
	} else if(n + tile_size_x == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + tile_size_x == fcols) ? 0 : n + tile_size_x;
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), 
		(uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), 
		(uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	if(!(CUR_SCRATCH[0] + params.stride[1] >= rows * params.stride[1] && 
		CUR_SCRATCH[1] + params.stride[2] >= cols * params.stride[2]))
		transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}