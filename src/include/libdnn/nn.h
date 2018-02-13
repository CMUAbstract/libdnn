#ifndef NN_H
#define NN_H
#include "types.h"

#define MAX_MAT_SIZE 0x100
#define MAX_LAYER_SIZE 0x3000
#define TASK_UID_NN_OFFSET 20

extern uint stride[3];
extern uint size[3];

void task_init_nn();
void task_d_conv();
void task_d_conv1d();
void task_s_conv();
void task_d_fc();
void task_s_fc();
void task_pool();
void task_relu();
void task_filter();

#endif