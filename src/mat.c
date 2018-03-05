#include <libio/console.h>
#include "mat.h" 
#include "misc.h"

uint mat_get_dim(mat_t *m, uint axis) {
	return m->dims[axis];
}

void mat_reshape(mat_t *m, uint len, uint dims[]) {
	m->len_dims = len;
	for(uint i = 0; i < len; i ++) {
		m->dims[i] = dims[i];
	}
}

mat_t mat_constrain(mat_t *m, uint len, uint idxs[]) {
	uint offset = 0;
	for(uint i = 0; i < len; i ++) {
		uint factor = 1;
		for(short j = i + 1; j < m->len_dims; j ++) {
			factor *= m->dims[j];
		}
		offset += factor * idxs[i];
	}
	mat_t c_m;
	uint j = 0;
	for(uint i = len; i < m->len_dims; i++) {
		c_m.dims[j] = m->dims[i];
		c_m.sparse.dims[j] = m->sparse.dims[i];
		j++;
	}
	j = 0;
	for(uint i = len; i < 10; i++) {
		c_m.sparse.dims[j] = m->sparse.dims[i];
		j++;
	}
	c_m.len_dims = m->len_dims - len;
	c_m.data = m->data + offset;
	c_m.sparse.offsets = m->sparse.offsets + offset;
	c_m.sparse.sizes = m->sparse.sizes;
	return c_m;
}

uint _offset_calc(void *_m, uint len, uint idxs[]) {
	mat_t *m = (mat_t *)_m;
	uint offset = 0;
	for(uint i = 0; i < len; i ++) {
		uint factor = 1;
		uint factor_idx = m->len_dims - i;
		for(short j = factor_idx - 1; j > 0; j --) {
			factor *= m->dims[j];
		}
		offset += factor * idxs[i];
	}
	return offset;
}

fixed mat_get(mat_t *m, uint len, uint idxs[]) {
	return *(m->data + _offset_calc(m, len, idxs));
}

void mat_set(mat_t *m, fixed val, uint len, uint idxs[]) {
	*(m->data + _offset_calc(m, len, idxs)) = val;
}

void mat_dump(mat_t *m, uint which) {
	uint rows = MAT_GET_DIM(m, m->len_dims - 2);
	uint cols = MAT_GET_DIM(m, m->len_dims - 1);
	PRINTF("\r\n===================== \r\n");
	PRINTF("\r\nRows: %u\r\n", rows);
	PRINTF("Cols: %u\r\n", cols);
	for(uint i = 0; i < rows; i ++) {
		for(uint j = 0; j < cols; j ++) {
			PRINTF("%i ", MAT_GET(m, which, i, j));
			if((j + 1) % cols == 0)
				PRINTF("\r\n");
		}
	}
	PRINTF("done ");
	PRINTF("===================== \r\n");
}

void mat_debug_dump(mat_t *m, uint which, fixed *dest) {
	fixed *dest_ptr = dest;
	uint rows = MAT_GET_DIM(m, m->len_dims - 2);
	uint cols = MAT_GET_DIM(m, m->len_dims - 1);
	for(uint i = 0; i < rows; i ++) {
		for(uint j = 0; j < cols; j ++) {
			*dest_ptr = MAT_GET(m, which, i, j);
			dest_ptr++;
		}
	}
}