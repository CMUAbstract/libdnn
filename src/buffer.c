#include "buffer.h"
#include "mem.h"
#include "types.h"
#include "mat.h"

__fram fixed mat_buffers[MAT_BUFFER_NUMBER][MAT_BUFFER_SIZE];
__hifram fixed layer_buffers[LAYER_BUFFER_NUMBER][LAYER_BUFFER_SIZE];