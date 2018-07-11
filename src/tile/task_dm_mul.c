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

TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_mul);

static __fram mat_t buf = {.data = LAYER_BUFFER(3)};
static __fram mat_t *inter = &buf;

// Dense scalar multiplication
void task_dm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(filter, 0);
	uint16_t cols = MAT_GET_DIM(filter, 1);
	uint16_t dcols = MAT_GET_DIM(dest, 1);
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_common_tile_size(rows, dcols, CONFIG_TILE_SIZE);

	MAT_RESHAPE(inter, tile_size_y, tile_size_x);

	for(uint16_t i = 0; i < tile_size_y; i++) {
		uint16_t idx_i = CUR_SCRATCH[0] + i;
		for(uint16_t k = 0; k < tile_size_y; k++) {
			uint16_t idx_k = CUR_SCRATCH[2] + k;
			fixed w = 0;
			for(uint16_t j = 0; j < tile_size_x; j++) {
				uint16_t idx_j = CUR_SCRATCH[1] + j;
				fixed tmp = F_MUL(MAT_GET(filter, idx_i, idx_j), 
					MAT_GET(src, idx_j, idx_k));
				w = F_ADD(w, tmp);
			}
			if(CUR_SCRATCH[1] >= tile_size_x) 
				w = F_ADD(w, MAT_GET(dest, idx_i, idx_k));
			MAT_SET(inter, w, i, k);
			uint16_t where = idx_i * dcols + idx_k;
			write_to_gbuf((uint8_t *)(inter->data + i * tile_size_x + k), 
				(uint8_t *)(dest->data + where), sizeof(fixed));
		}
	}

	uint16_t i = CUR_SCRATCH[0];
	uint16_t j = CUR_SCRATCH[1];
	uint16_t k = CUR_SCRATCH[2];
	// i
	scratch_bak[0] = CUR_SCRATCH[0];
	if(CUR_SCRATCH[1] + tile_size_x == cols) {
		scratch_bak[0] = CUR_SCRATCH[0] + tile_size_y;
	}
	// j
	scratch_bak[1] = CUR_SCRATCH[1];
	if(CUR_SCRATCH[1] + tile_size_x == cols && 
		CUR_SCRATCH[2] + tile_size_y == dcols) {
		scratch_bak[1] = 0;
	} else if(CUR_SCRATCH[2] + tile_size_y == dcols) {
		scratch_bak[1] = CUR_SCRATCH[1] + tile_size_x;
	}
	// k
	scratch_bak[2] = CUR_SCRATCH[2] + tile_size_y;
	if(CUR_SCRATCH[2] + tile_size_y == dcols) {
		scratch_bak[2] = 0;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), 
		(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	if(!(CUR_SCRATCH[0] + tile_size_y == rows && 
		CUR_SCRATCH[1] + tile_size_x == cols && 
		CUR_SCRATCH[2] + tile_size_y == dcols)) {
		transition_to(CUR_TASK);	
	}
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}