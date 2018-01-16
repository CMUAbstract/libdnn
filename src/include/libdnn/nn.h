#ifndef NN_H
#define NN_H

#define MAX_MAT_SIZE 0x100
#define MAX_LAYER_SIZE 0x3000
#define TASK_UID_NN_OFFSET 20

void task_init_nn();
void task_d_conv();
void task_s_conv();
void task_d_fc();
void task_s_fc();
void task_pool();
void task_relu();
#endif