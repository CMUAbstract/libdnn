#include <string.h>
#include <msp430.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "linalg.h"
#include "nonlinear.h"
#include "buffer.h"
#include "blas.h"
#include "mem.h"
#include "state.h"
#include "misc.h"
#include "cleanup.h"

static __fram mat_t m = {.data = LAYER_BUFFER(0)};
static __fram mat_t *inter = &m;

// Public tasks
TASK(TASK_UID_LINALG_OFFSET, task_norm);

void task_norm() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	MAT_RESHAPE(inter, 1, 1);
	if(CUR_SCRATCH[0] == 0) {
		PRINTF("\r\n    Taking transpose");
		// Assumes dest, src in that order
		MAT_RESHAPE(dest, MAT_GET_DIM(dest, 1), MAT_GET_DIM(dest, 0));
		PUSH_STACK(mat_stack, dest, src);
		scratch_bak[0] = 1;	
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_transpose)->info.return_task = CUR_TASK;
		TRANSITION_TO(task_transpose);
	} else if(CUR_SCRATCH[0] == 1) {
		PRINTF("\r\n    Finding norm");
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, dest, inter, src);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_dm_mul)->info.return_task = CUR_TASK;
		TRANSITION_TO(task_dm_mul);
	} else if(CUR_SCRATCH[0] == 2) {
		PRINTF("\r\n    Taking sqrt");
		scratch_bak[0] = 3;
		scratch_bak[1] = F_SQRT(MAT_GET(inter, 0, 0)); 
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(inter->data), sizeof(fixed));
		transition_to(CUR_TASK);
	} else if(CUR_SCRATCH[0] == 3) {
		PRINTF("\r\n    Applying norm");
		// Assumes filter, dest, src in that order
		MAT_RESHAPE(dest, MAT_GET_DIM(dest, 1), MAT_GET_DIM(dest, 0));
		PUSH_STACK(mat_stack, inter, dest, src);
		scratch_bak[0] = 4;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TASK_REF(task_ds_div)->info.return_task = CUR_TASK;
		TRANSITION_TO(task_ds_div);
	}
	POP_STACK(mat_stack, 2);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}