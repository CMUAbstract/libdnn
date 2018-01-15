#ifndef TYPES_H
#define TYPES_H

typedef signed int fixed;
typedef unsigned int uint;
typedef unsigned int uchar;

typedef struct {
	uint dims[10];
	uint len_dims;
	fixed *data;
	// To support higher dimensional sparse matrices
	fixed *sparse_sizes;
	uint sparse_dims[10];
} mat_t;

#endif