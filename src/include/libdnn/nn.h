#ifndef NN_H
#define NN_H
#include "types.h"

#define TASK_UID_NN_OFFSET 20

extern bool same_padding;
extern uint stride[3];
extern uint size[3];

void task_d_conv();
void task_d_conv1d();
void task_s_conv();
void task_d_fc();
void task_s_fc();
void task_pool();
void task_relu();
void task_filter();

extern TASK_DEC(task_d_conv);
extern TASK_DEC(task_d_conv1d);
extern TASK_DEC(task_s_conv);
extern TASK_DEC(task_d_fc);
extern TASK_DEC(task_s_fc);
extern TASK_DEC(task_pool);
extern TASK_DEC(task_relu);
extern TASK_DEC(task_filter);

#endif