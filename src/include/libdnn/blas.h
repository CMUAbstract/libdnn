#ifndef BLAS_H
#define BLAS_H
#include <libalpaca/alpaca.h>
#include "types.h"

#define TASK_UID_BLAS_OFFSET 10

extern uint stride[3];

void task_ds_zero();
void task_ds_add();
void task_ds_mul();
void task_dm_add();
void task_dm_mul();
void task_dm_conv();
void task_dm_conv_same();
void task_dm_conv1d();
void task_sm_mul();
void task_sm_conv();
void task_sm_conv_same();

extern TASK_DEC(task_ds_zero);
extern TASK_DEC(task_ds_add);
extern TASK_DEC(task_ds_mul);
extern TASK_DEC(task_dm_add);
extern TASK_DEC(task_dm_mul);
extern TASK_DEC(task_dm_conv);
extern TASK_DEC(task_dm_conv_same);
extern TASK_DEC(task_sm_mul);
extern TASK_DEC(task_sm_conv);
extern TASK_DEC(task_sm_conv_same);

#endif