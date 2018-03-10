#ifndef BUFFER_H
#define BUFFER_H
#include <libalpaca/alpaca.h>
#include "types.h"

#define MAT_BUFFER_NUMBER 3
#define LAYER_BUFFER_NUMBER 3

#define MAT_BUFFER_SIZE 0x240
#define LAYER_BUFFER_SIZE 0x3000

#define TASK_UID_INIT_OFFSET 40

extern fixed mat_buffers[MAT_BUFFER_NUMBER][MAT_BUFFER_SIZE];
extern fixed layer_buffers[LAYER_BUFFER_NUMBER][LAYER_BUFFER_SIZE];

void task_init();

extern TASK_DEC(task_init);
#endif