#ifndef TYPES_H
#define TYPES_H

typedef unsigned int uint;

#if CONFIG_BITWIDTH == 8
typedef signed char fixed;
typedef unsigned char uchar;
#else
typedef signed int fixed;
typedef unsigned int uchar;
#endif

typedef struct {
	uint dims[10];
	uint len_dims;
	fixed *data;
	// To support higher dimensional sparse matrices
	struct {
		uint dims[10];
		uint len_dims;
		uint *offsets;
		uint *sizes;
	} sparse;
} mat_t;

#endif