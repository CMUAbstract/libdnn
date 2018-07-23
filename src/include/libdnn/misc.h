#ifndef MISC_H
#define MISC_H

#include <stdint.h>
#include <stdbool.h>

#ifndef CONFIG_CONSOLE
	#define printf(fmt, ...) (void)0
#endif

typedef struct {
	bool same_padding;
	bool transpose;
	uint16_t stride[3];
	uint16_t size[3];
} param_t;

extern param_t params;

#endif
