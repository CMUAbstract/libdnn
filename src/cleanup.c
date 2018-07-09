#include "cleanup.h"
#include "mem.h"

// Private tasks
TASK(404, task_cleanup);

// Resets a task
static __fram task_t *last_task;
void task_cleanup() {
	// PRINTF("\r\n     Finishing BLAS");
	memset(last_task->info.scratch, 0, sizeof(uint16_t) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

void setup_cleanup(task_t *task) {
	last_task = task;
}