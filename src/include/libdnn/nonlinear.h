#ifndef NONLINEAR_H
#define NONLINEAR_H
#include <libalpaca/alpaca.h>
#include "types.h"

#define TASK_UID_NONLINEAR_OFFSET 60

extern uint stride[3];
extern uint size[3];

void task_pool();
void task_relu();
void task_filter();
void task_transpose();

extern TASK_DEC(task_pool);
extern TASK_DEC(task_relu);
extern TASK_DEC(task_filter);
extern TASK_DEC(task_transpose);

#endif