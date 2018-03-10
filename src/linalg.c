#include <string.h>
#include <msp430.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>

#include "linalg.h"
#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "fixed.h"
#include "mat.h"
#include "misc.h"

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
	#pragma GCC warning "norm not yet implemented"
}
