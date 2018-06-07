#include <libfixed/fixed.h>
#include "buffer.h"
#include "mem.h"

__hifram fixed mat_buffers[MAT_BUF_NUMBER][CONFIG_MAT_BUF_SIZE];
__hifram fixed layer_buffers[LAYER_BUF_NUMBER][CONFIG_LAYER_BUF_SIZE];