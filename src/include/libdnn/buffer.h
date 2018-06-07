#ifndef BUFFER_H
#define BUFFER_H
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>

#define MAT_BUF_NUMBER 3
#define LAYER_BUF_NUMBER 3

#define TASK_UID_INIT_OFFSET 40

extern fixed mat_buffers[MAT_BUF_NUMBER][CONFIG_MAT_BUF_SIZE];
extern fixed layer_buffers[LAYER_BUF_NUMBER][CONFIG_LAYER_BUF_SIZE];

#define MAT_BUFFER(idx) (mat_buffers[idx])
#define LAYER_BUFFER(idx) (layer_buffers[idx])
#endif