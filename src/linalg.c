#include <string.h>
#include <msp430.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>

#include "linalg.h"
#include "nonlinear.h"
#include "buffer.h"
#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "fixed.h"
#include "mat.h"
#include "misc.h"

static __fram mat_t m = {.data = LAYER_BUFFER(0)};
static __fram mat_t *inter = &m;
static __fram uint scratch_bak[SCRATCH_SIZE];

// Public tasks
TASK(TASK_UID_LINALG_OFFSET, task_norm);

// Private tasks
void task_cleanup_linalg();
TASK(TASK_UID_LINALG_OFFSET + 1, task_cleanup_linalg);

// Resets a task
static __fram task_t *last_task;
void task_cleanup_linalg() {
	PRINTF("\r\n Cleaning up Linalg");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

void task_norm() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	MAT_RESHAPE(inter, 1, 1);
	if(CUR_INFO.scratch[0] == 0) {
		PRINTF("\r\n    Taking transpose");
		// Assumes dest, src in that order
		MAT_RESHAPE(dest, MAT_GET_DIM(dest, 1), MAT_GET_DIM(dest, 0));
		PUSH_STACK(mat_stack, dest, src);
		scratch_bak[0] = 1;	
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		TASK_REF(task_transpose)->info.return_task = CUR_TASK;
		TRANSITION_TO(task_transpose);
	} else if(CUR_INFO.scratch[0] == 1) {
		PRINTF("\r\n    Finding norm");
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, dest, inter, src);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		TASK_REF(task_dm_mul)->info.return_task = CUR_TASK;
		TRANSITION_TO(task_dm_mul);
	} else if(CUR_INFO.scratch[0] == 2) {
		PRINTF("\r\n    Taking sqrt");
		scratch_bak[0] = 3;
		scratch_bak[1] = F_SQRT(MAT_GET(inter, 0, 0)); 
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(inter->data), sizeof(fixed));
		transition_to(CUR_TASK);
	} else if(CUR_INFO.scratch[0] == 3) {
		PRINTF("\r\n    Applying norm");
		// Assumes filter, dest, src in that order
		MAT_RESHAPE(dest, MAT_GET_DIM(dest, 1), MAT_GET_DIM(dest, 0));
		PUSH_STACK(mat_stack, inter, dest, src);
		scratch_bak[0] = 4;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		TASK_REF(task_ds_div)->info.return_task = CUR_TASK;
		TRANSITION_TO(task_ds_div);
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 2);
	TRANSITION_TO(task_cleanup_linalg);
}