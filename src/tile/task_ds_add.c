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
#include "tile.h"

TASK(TASK_UID_BLAS_OFFSET + 1, task_ds_add);

// Dense scalar addition
void task_ds_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint16_t rows = MAT_GET_DIM(src, 0);
	uint16_t cols = MAT_GET_DIM(src, 1);
	uint16_t tile_size_x = greatest_tile_size(cols, CONFIG_TILE_SIZE);
	uint16_t tile_size_y = greatest_tile_size(rows, CONFIG_TILE_SIZE);
	for(uint16_t i = CUR_SCRATCH[0]; i < CUR_SCRATCH[0] + tile_size_y; i++) {
		for(uint16_t j = CUR_SCRATCH[1]; j < CUR_SCRATCH[1] + tile_size_x; j++) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
	}
	uint16_t j = CUR_SCRATCH[1];
	scratch_bak[0] = CUR_SCRATCH[0];
	scratch_bak[1] = CUR_SCRATCH[1] + tile_size_x;
	if(j + tile_size_x == cols) {
		scratch_bak[0] = CUR_SCRATCH[0] + tile_size_y;
		scratch_bak[1] = 0;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), 
		(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), 
		(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	if(!(CUR_SCRATCH[0] + tile_size_y == rows 
		&& CUR_SCRATCH[1] + tile_size_x ==  cols)) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
