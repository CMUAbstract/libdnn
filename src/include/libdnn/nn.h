#ifndef NN_H
#define NN_H
#include <libalpaca/alpaca.h>
#include "types.h"

#define TASK_UID_NN_OFFSET 20

extern bool same_padding;

void task_d_conv();
void task_s_conv();
void task_d_fc();
void task_s_fc();

extern TASK_DEC(task_d_conv);
extern TASK_DEC(task_s_conv);
extern TASK_DEC(task_d_fc);
extern TASK_DEC(task_s_fc);

#endif