#ifndef BLAS_H
#define BLAS_H
#include <libalpaca/alpaca.h>
#include "types.h"

#define TASK_UID_BLAS_OFFSET 10
#define MAX_MAT_SIZE 0x240
#define MAX_TILE_SIZE 12

extern uint stride[3];

extern task_t _task_task_sm_mul;
void task_init_blas();
void task_ds_zero();
void task_ds_add();
void task_dm_add();
void task_dm_mul();
void task_dm_conv();
void task_dm_conv1d();
void task_sm_mul();
void task_sm_conv();

#endif