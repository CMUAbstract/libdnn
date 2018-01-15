#ifndef MAT_H
#define MAT_H

#include "types.h"
#include "misc.h"

#define MAT_NUMARGS(...)  (sizeof((uint[]){__VA_ARGS__})/sizeof(uint))

#define MAT_RESHAPE(m, ...) (mat_reshape(m, MAT_NUMARGS(__VA_ARGS__), 			\
								(uint[]){__VA_ARGS__}))

#define MAT_CONSTRAIN(m, ...) (mat_constrain(m, MAT_NUMARGS(__VA_ARGS__), 		\
								(uint[]){__VA_ARGS__}))

#define MAT_GET(m, ...) (													\
	(MAT_NUMARGS(__VA_ARGS__) == 1) ? 											\
		*(m->data + ((uint[]){__VA_ARGS__})[0]) :							\
	(MAT_NUMARGS(__VA_ARGS__) == 2) ? 											\
		*(m->data + ((uint[]){__VA_ARGS__})[0] * 							\
			m->dims[m->len_dims - 1] + ((uint[]){__VA_ARGS__})[1]):			\
	(MAT_NUMARGS(__VA_ARGS__) == 3) ? 											\
		*(m->data + ((uint[]){__VA_ARGS__})[0] *							\
			m->dims[m->len_dims - 1] * m->dims[m->len_dims - 2] +			\
			m->dims[m->len_dims - 1] * ((uint[]){__VA_ARGS__})[1] + 		\
			((uint[]){__VA_ARGS__})[2]):									\
	mat_get(m, MAT_NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))

#define MAT_SET(m, val, ...) (												\
	(MAT_NUMARGS(__VA_ARGS__) == 1) ? 											\
		*(m->data +	((uint[]){__VA_ARGS__})[0]) = val :						\
	(MAT_NUMARGS(__VA_ARGS__) == 2) ? 											\
		*(m->data + ((uint[]){__VA_ARGS__})[0] * 							\
			m->dims[m->len_dims - 1] + ((uint[]){__VA_ARGS__})[1]) = val :	\
	(MAT_NUMARGS(__VA_ARGS__) == 3) ? 											\
		*(m->data + ((uint[]){__VA_ARGS__})[0] *							\
			m->dims[m->len_dims - 1] * m->dims[m->len_dims - 2] +			\
			m->dims[m->len_dims - 1] * ((uint[]){__VA_ARGS__})[1] + 		\
			((uint[]){__VA_ARGS__})[2]) = val :								\
	mat_set(m, val, MAT_NUMARGS(__VA_ARGS__), (uint[]){__VA_ARGS__}))

#define MAT_GET_DIM(m, axis) (mat_get_dim(m, axis))

#ifndef INTERMITTENT
	#define MAT_DUMP(m, w) (mat_dump(m, w))
#else
	#define MAT_DUMP(m, w) (0)
#endif

#define MAT_DEBUG_DUMP(m, v, d) (mat_debug_dump(m, v, d))

uint mat_get_dim(mat_t *, uint axis);

void mat_reshape(mat_t *, uint, uint[]);

mat_t mat_constrain(mat_t *, uint, uint[]);

fixed mat_get(mat_t *, uint, uint[]);

void mat_set(mat_t *, fixed, uint, uint[]);

void mat_dump(mat_t *, uint);

void mat_debug_dump(mat_t *, uint, fixed *);

#endif