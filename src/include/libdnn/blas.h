#ifndef BLAS_H
#define BLAS_H

#include <stdbool.h>

#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>

#if CONFIG_DMA == 0 // Disable DMA
#pragma message "Disable DMA"
	#define DMA_ENABLE && 0
#elif CONFIG_DMA == 1 // Always use DMA
#pragma message "Always DMA"
	#define DMA_ENABLE || 1
#else // Choose DMA
#pragma message "Choose DMA"
	#define DMA_ENABLE && 1
#endif

#define TASK_UID_BLAS_OFFSET 10
#define SHIFT 5

void task_ds_zero();
void task_ds_add();
void task_ds_mul();
void task_ds_div();
void task_dm_add();
void task_dm_mul();
void task_dm_conv();
void task_sm_mul();
void task_svm_mul();
void task_sm_conv();

extern TASK_DEC(task_ds_zero);
extern TASK_DEC(task_ds_add);
extern TASK_DEC(task_ds_mul);
extern TASK_DEC(task_ds_div);
extern TASK_DEC(task_dm_add);
extern TASK_DEC(task_dm_mul);
extern TASK_DEC(task_dm_conv);
extern TASK_DEC(task_sm_mul);
extern TASK_DEC(task_svm_mul);
extern TASK_DEC(task_sm_conv);

#endif