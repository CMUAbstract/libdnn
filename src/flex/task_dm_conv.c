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
TASK(TASK_UID_BLAS_OFFSET + 7, task_dm_conv_same);

// Dense matrix convolution
void task_dm_conv() {
	/*mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint16_t k = CUR_SCRATCH[2];
	uint16_t l = CUR_SCRATCH[3];
	uint16_t n = CUR_SCRATCH[4];
	mat_t *prev_dest = (CUR_SCRATCH[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_SCRATCH[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
		}
	} else {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k;
	scratch_bak[3] = l;
	if(n + 1 == fcols && l + 1 == frows) {
		scratch_bak[3] = 0;
		scratch_bak[2] = k + 1;
	} else if(n + 1 == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + 1 == fcols) ? 0 : n + 1;
	scratch_bak[5] = ~CUR_SCRATCH[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_SCRATCH + 5), sizeof(uint16_t));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);*/
}

// Dense matrix convolution
void task_dm_conv_same() {
	/*mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint16_t rows = MAT_GET_DIM(dest, 0);
	uint16_t cols = MAT_GET_DIM(dest, 1);

	uint16_t flayers = MAT_GET_DIM(filter, 0);
	uint16_t frows = MAT_GET_DIM(filter, 1);
	uint16_t fcols = MAT_GET_DIM(filter, 2);
	MAT_RESHAPE(inter1, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint16_t k = CUR_SCRATCH[2];
	uint16_t l = CUR_SCRATCH[3];
	uint16_t n = CUR_SCRATCH[4];
	mat_t *prev_dest = (CUR_SCRATCH[5] % 2 == 0) ? inter2 : inter1;
	if(k < flayers && l < frows && n < fcols) {
		dest = (CUR_SCRATCH[5] % 2 == 0) ? inter1 : inter2;
	}

	if(k | l | n) {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, k, l, n), MAT_GET(src, k, i + l, j + n));
				if(i + l >= MAT_GET_DIM(src, 1) || j + n >= MAT_GET_DIM(src, 2)) {
					w = 0;
				}
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
		}
	} else {
		for(uint16_t i = CUR_SCRATCH[0]; i < rows; i = ++CUR_SCRATCH[0]) {
			for(uint16_t j = CUR_SCRATCH[1]; j < cols; j = ++CUR_SCRATCH[1]) {
				fixed w = F_MUL(MAT_GET(filter, 0, 0, 0), MAT_GET(src, 0, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_SCRATCH[1] = 0;
		}
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = k;
	scratch_bak[3] = l;
	if(n + 1 == fcols && l + 1 == frows) {
		scratch_bak[3] = 0;
		scratch_bak[2] = k + 1;
	} else if(n + 1 == fcols) {
		scratch_bak[3] = l + 1;
	}
	scratch_bak[4] = (n + 1 == fcols) ? 0 : n + 1;
	scratch_bak[5] = ~CUR_SCRATCH[5];
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_SCRATCH + 3), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_SCRATCH + 4), sizeof(uint16_t));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_SCRATCH + 5), sizeof(uint16_t));
	if(k < flayers && l < frows && n < fcols) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup); */
}