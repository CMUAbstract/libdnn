#include "buffer.h"
#include "mem.h"
#include "types.h"

__hifram fixed mat_buffers[MAT_BUFFER_NUMBER][CONFIG_MAT_BUF_SIZE];
__hifram fixed layer_buffers[LAYER_BUFFER_NUMBER][CONFIG_LAYER_BUF_SIZE];