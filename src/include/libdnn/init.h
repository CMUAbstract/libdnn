#ifndef INIT_H
#define INIT_H
#include <libalpaca/alpaca.h>
#include "types.h"

#define SMALL_BUFFER_NUMBER 3
#define LARGE_BUFFER_NUMBER 3

#define SMALL_BUFFER_SIZE 0x240
#define LARGE_BUFFER_SIZE 0x3000

extern fixed small_buffers[3][MAX_MAT_SIZE];
extern fixed large_buffers[3][MAX_LAYER_SIZE];

#endif