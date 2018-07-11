#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>
#include <libmspdriver/driverlib.h>
#include <libdsp/DSPLib.h>

#include "lea.h"
#include "mem.h"
#include "blas.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"
#include "profile.h"
#include "cleanup.h"

TASK(TASK_UID_BLAS_OFFSET + 10, task_sm_conv);

// Dense matrix multiplication
void task_sm_conv() {
	uint16_t tile_size = check_calibrate();
	POP_STACK(mat_stack, 3);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}