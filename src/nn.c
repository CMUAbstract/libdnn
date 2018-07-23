#include <stdbool.h>
#include <string.h>
#include <msp430.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libfixed/fixed.h>
#include <libmat/mat.h>

#include "nn.h"
#include "blas.h"
#include "mem.h"
#include "state.h"
#include "buffer.h"
#include "misc.h"
#include "cleanup.h"

static __fram mat_t m = {.data = LAYER_BUFFER(0)};
static __fram mat_t *inter = &m;
static __fram mat_t c_src, c_filter, c_dest, c_inter;
static __fram mat_t *c_filter_ptr = &c_filter;
static __fram mat_t *c_src_ptr = &c_src;
static __fram mat_t *c_dest_ptr = &c_dest;
static __fram mat_t *c_inter_ptr = &c_inter;

// Public tasks
TASK(TASK_UID_NN_OFFSET + 0, task_d_conv);
TASK(TASK_UID_NN_OFFSET + 1, task_d_depthconv);
TASK(TASK_UID_NN_OFFSET + 2, task_s_conv);
TASK(TASK_UID_NN_OFFSET + 3, task_s_depthconv);
TASK(TASK_UID_NN_OFFSET + 4, task_d_fc);
TASK(TASK_UID_NN_OFFSET + 5, task_s_fc);

#ifdef CONFIG_LEA
#pragma message "Using LEA Backend"
void task_d_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = MAT_GET_DIM(w, 0);
	if(CUR_SCRATCH[0] == 0) { // Sparse Convolve
		PRINTF("\r\n Shifting src");
		mat_reshape(inter, src->dims, src->len_dims);
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					MAT_SET(inter, (MAT_GET(src, k, i, j) << SHIFT), k, i, j);
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 1;
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 1) { // Sparse Convolve
		PRINTF("\r\n Writing back");
		mat_reshape(inter, src->dims, src->len_dims);
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					MAT_SET(src, MAT_GET(inter, k, i, j), k, i, j);
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 2;
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 2) {
		uint16_t i = CUR_SCRATCH[1];
		if(i < filters) {
			PRINTF("\r\n    Convolving %u", i);
			TASK_REF(task_dm_conv)->info.return_task = CUR_TASK;
			// Assumes filter, dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			c_filter = MAT_CONSTRAIN(w, i);
			PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), 
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_dm_conv);
		}
		scratch_bak[0] = 3;	
		scratch_bak[1] = 0;	
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_d_depthconv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = MAT_GET_DIM(w, 0);
	if(CUR_SCRATCH[0] == 0) { // Sparse Convolve
		PRINTF("\r\n Shifting src");
		mat_reshape(inter, src->dims, src->len_dims);
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					MAT_SET(inter, (MAT_GET(src, k, i, j) << SHIFT), k, i, j);
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 1;
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 1) {
		PRINTF("\r\n Writing back");
		mat_reshape(inter, src->dims, src->len_dims);
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					MAT_SET(src, MAT_GET(inter, k, i, j), k, i, j);
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 2;
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 2) {
		uint16_t i = CUR_SCRATCH[1];
		PRINTF("\r\n    Convolving %u", i);
		if(i < filters) {
			TASK_REF(task_dm_conv)->info.return_task = CUR_TASK;
			// Assumes filter, dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			c_filter = MAT_CONSTRAIN(w, i);
			c_src = MAT_CONSTRAIN(src, i);
			MAT_RESHAPE(c_src_ptr, 1, MAT_GET_DIM(src, 1), MAT_GET_DIM(src, 2));
			PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, c_src_ptr);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), 
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_dm_conv);
		}
		scratch_bak[0] = 3;	
		scratch_bak[1] = 0;	
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

static __fram bool transpose = false;
static __fram mat_t src_bak;
static __fram mat_t *src_bak_ptr = &src_bak;
void task_s_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = w->sparse.dims[0];
	transpose = (w->sparse.dims[2] > 1 && w->sparse.dims[3] == 1);
	if(CUR_SCRATCH[0] == 0) { // Sparse Convolve
		PRINTF("\r\n Shifting src");
		mat_reshape(inter, src->dims, src->len_dims);
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					if(MAT_GET_DIM(src, 1) == 1) {
						MAT_SET(inter, MAT_GET(src, k, i, j), k, i, j);
					} else {
						MAT_SET(inter, (MAT_GET(src, k, i, j) << SHIFT), k, i, j);
					}
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 1;
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 1) { // Sparse Convolve
		PRINTF("\r\n Writing back");
		mat_reshape(inter, src->dims, src->len_dims);
		if(transpose) {
			mat_copy(src, src_bak_ptr);
			MAT_RESHAPE(src_bak_ptr, MAT_GET_DIM(src, 0), 
				MAT_GET_DIM(src, 2), MAT_GET_DIM(src, 1));
		}
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					if(transpose) {
						MAT_SET(src_bak_ptr, MAT_GET(inter, k, i, j), k, j, i);
					} else {
						MAT_SET(src, MAT_GET(inter, k, i, j), k, i, j);
					}
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 2;
		scratch_bak[2] = 0;
		if(transpose) {
			PRINTF("\r\n Taking transpose");
			write_to_gbuf((uint8_t *)(src_bak_ptr), 
				(uint8_t *)(src), sizeof(mat_t));
		}
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 2) {
		uint16_t i = CUR_SCRATCH[1];
		uint16_t running_size = CUR_SCRATCH[2];
		params.transpose = transpose;
		if(i < filters) {
			if(w->sparse.sizes[i] > 0) {
				PRINTF("\r\n     Convolving %u %u %u", 
					i, running_size, w->sparse.sizes[i]);
				TASK_REF(task_sm_conv)->info.return_task = CUR_TASK;
				// Assumes filter, dest, src in that order
				c_filter = MAT_CONSTRAIN(w, running_size);
				c_filter.dims[0] = w->sparse.sizes[i];
				c_filter.sparse.len_dims = w->sparse.len_dims - 1;
				c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
				PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, src);
				scratch_bak[1] = i + 1;
				scratch_bak[2] = running_size + w->sparse.sizes[i];
				write_to_gbuf((uint8_t *)(scratch_bak + 1), 
					(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
				write_to_gbuf((uint8_t *)(scratch_bak + 2), 
					(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
				TRANSITION_TO(task_sm_conv);
			}
			PRINTF("\r\n     Zeroing %u", i);
			TASK_REF(task_ds_zero)->info.return_task = CUR_TASK;
			// Assumes dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			PUSH_STACK(mat_stack, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), 
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_ds_zero);
		}
		// All done
		scratch_bak[0] = 3;	
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		params.transpose = false;
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 

			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	params.transpose = false;
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_s_depthconv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = w->sparse.dims[0];
	transpose = (w->sparse.dims[2] > 1 && w->sparse.dims[3] == 1);
	if(CUR_SCRATCH[0] == 0) { // Sparse Convolve
		PRINTF("\r\n Shifting src %u %u %u", filters, w->sparse.dims[1], w->sparse.dims[2]);
		mat_reshape(inter, src->dims, src->len_dims);
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					MAT_SET(inter, (MAT_GET(src, k, i, j) << SHIFT), k, i, j);
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 1;
		scratch_bak[2] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 1) { // Sparse Convolve
		PRINTF("\r\n Writing back");
		mat_reshape(inter, src->dims, src->len_dims);
		if(transpose) {
			mat_copy(src, src_bak_ptr);
			MAT_RESHAPE(src_bak_ptr, MAT_GET_DIM(src, 0), 
				MAT_GET_DIM(src, 2), MAT_GET_DIM(src, 1));
		}
		for(uint16_t k = CUR_SCRATCH[2]; 
			k < MAT_GET_DIM(src, 0); k = ++CUR_SCRATCH[2]) {
			for(uint16_t i = CUR_SCRATCH[3]; 
				i < MAT_GET_DIM(src, 1); i = ++CUR_SCRATCH[3]) {
				for(uint16_t j = CUR_SCRATCH[4]; 
					j < MAT_GET_DIM(src, 2); j = ++CUR_SCRATCH[4]) {
					if(transpose) {
						MAT_SET(src_bak_ptr, MAT_GET(inter, k, i, j), k, j, i);
					} else {
						MAT_SET(src, MAT_GET(inter, k, i, j), k, i, j);
					}
				}
				CUR_SCRATCH[4] = 0;
			}
			CUR_SCRATCH[3] = 0;
		}
		scratch_bak[0] = 2;
		scratch_bak[2] = 0;
		if(transpose) {
			PRINTF("\r\n Taking transpose");
			write_to_gbuf((uint8_t *)(src_bak_ptr), 
				(uint8_t *)(src), sizeof(mat_t));
		}
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));	
		write_to_gbuf((uint8_t *)(scratch_bak + 2), 
			(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));	
		transition_to(CUR_TASK);	
	} else if(CUR_SCRATCH[0] == 2) {
		uint16_t i = CUR_SCRATCH[1];
		uint16_t running_size = CUR_SCRATCH[2];
		params.transpose = transpose;
		if(i < filters) {
			if(w->sparse.sizes[i] > 0) {
				PRINTF("\r\n     Convolving %u %u %u",
					i, running_size, w->sparse.sizes[i]);
				TASK_REF(task_sm_conv)->info.return_task = CUR_TASK;
				// Assumes filter, dest, src in that order
				c_filter = MAT_CONSTRAIN(w, running_size);
				c_filter.dims[0] = w->sparse.sizes[i];
				c_filter.sparse.len_dims = w->sparse.len_dims - 1;
				c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
				c_src = MAT_CONSTRAIN(src, i);
				MAT_RESHAPE(c_src_ptr, 1, MAT_GET_DIM(src, 1), MAT_GET_DIM(src, 2));
				PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, c_src_ptr);
				scratch_bak[1] = i + 1;
				scratch_bak[2] = running_size + w->sparse.sizes[i];
				write_to_gbuf((uint8_t *)(scratch_bak + 1), 
					(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
				write_to_gbuf((uint8_t *)(scratch_bak + 2), 
					(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
				TRANSITION_TO(task_sm_conv);
			}
			PRINTF("\r\n     Zeroing %u", i);
			TASK_REF(task_ds_zero)->info.return_task = CUR_TASK;
			// Assumes dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			PUSH_STACK(mat_stack, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1),
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_ds_zero);
		}
		// All done
		scratch_bak[0] = 3;	
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak),
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1),
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		params.transpose = false;
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1),
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	params.transpose = false;
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
#else
void task_d_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = MAT_GET_DIM(w, 0);
	if(CUR_SCRATCH[0] == 0) { // Do convolution on all filters
		uint16_t i = CUR_SCRATCH[1];
		if(i < filters) {
			PRINTF("\r\n    Convolving %u", i);
			TASK_REF(task_dm_conv)->info.return_task = CUR_TASK;
			// Assumes filter, dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			c_filter = MAT_CONSTRAIN(w, i);
			PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), 
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_dm_conv);
		}
		scratch_bak[0] = 1;	
		scratch_bak[1] = 0;	
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_d_depthconv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = MAT_GET_DIM(w, 0);
	if(CUR_SCRATCH[0] == 0) { // Do convolution on all filters
		uint16_t i = CUR_SCRATCH[1];
		PRINTF("\r\n    Convolving %u", i);
		if(i < filters) {
			TASK_REF(task_dm_conv)->info.return_task = CUR_TASK;
			// Assumes filter, dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			c_filter = MAT_CONSTRAIN(w, i);
			c_src = MAT_CONSTRAIN(src, i);
			MAT_RESHAPE(c_src_ptr, 1, MAT_GET_DIM(src, 1), MAT_GET_DIM(src, 2));
			PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, c_src_ptr);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), 
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_dm_conv);
		}
		scratch_bak[0] = 1;	
		scratch_bak[1] = 0;	
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_s_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = w->sparse.dims[0];
	if(CUR_SCRATCH[0] == 0) { // Sparse Convolve
		uint16_t i = CUR_SCRATCH[1];
		uint16_t running_size = CUR_SCRATCH[2];
		if(i < filters) {
			if(w->sparse.sizes[i] > 0) {
				PRINTF("\r\n     Convolving %u %u %u",
					i, running_size, w->sparse.sizes[i]);
				TASK_REF(task_sm_conv)->info.return_task = CUR_TASK;
				// Assumes filter, dest, src in that order
				c_filter = MAT_CONSTRAIN(w, running_size);
				c_filter.dims[0] = w->sparse.sizes[i];
				c_filter.sparse.len_dims = w->sparse.len_dims - 1;
				c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
				PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, src);
				scratch_bak[1] = i + 1;
				scratch_bak[2] = running_size + w->sparse.sizes[i];
				write_to_gbuf((uint8_t *)(scratch_bak + 1), 
					(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
				write_to_gbuf((uint8_t *)(scratch_bak + 2), 
					(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
				TRANSITION_TO(task_sm_conv);
			}
			PRINTF("\r\n     Zeroing %u", i);
			TASK_REF(task_ds_zero)->info.return_task = CUR_TASK;
			// Assumes dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			PUSH_STACK(mat_stack, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), 
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_ds_zero);
		}
		// All done
		scratch_bak[0] = 1;	
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_s_depthconv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	uint16_t filters = w->sparse.dims[0];
	if(CUR_SCRATCH[0] == 0) { // Sparse Convolve
		uint16_t i = CUR_SCRATCH[1];
		uint16_t running_size = CUR_SCRATCH[2];
		if(i < filters) {
			if(w->sparse.sizes[i] > 0) {
				PRINTF("\r\n     Convolving %u %u %u",
					i, running_size, w->sparse.sizes[i]);
				TASK_REF(task_sm_conv)->info.return_task = CUR_TASK;
				// Assumes filter, dest, src in that order
				c_filter = MAT_CONSTRAIN(w, running_size);
				c_filter.dims[0] = w->sparse.sizes[i];
				c_filter.sparse.len_dims = w->sparse.len_dims - 1;
				c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
				c_src = MAT_CONSTRAIN(src, i); // PROBLEM HERE because expect 3D Tensor
				MAT_RESHAPE(c_src_ptr, 1, MAT_GET_DIM(src, 1), MAT_GET_DIM(src, 2));
				PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, c_src_ptr);
				scratch_bak[1] = i + 1;
				scratch_bak[2] = running_size + w->sparse.sizes[i];
				write_to_gbuf((uint8_t *)(scratch_bak + 1), 
					(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
				write_to_gbuf((uint8_t *)(scratch_bak + 2), 
					(uint8_t *)(CUR_SCRATCH + 2), sizeof(uint16_t));
				TRANSITION_TO(task_sm_conv);
			}
			PRINTF("\r\n     Zeroing %u", i);
			TASK_REF(task_ds_zero)->info.return_task = CUR_TASK;
			// Assumes dest, src in that order
			c_inter = (b == NULL) ? MAT_CONSTRAIN(dest, i) :  MAT_CONSTRAIN(inter, i);
			PUSH_STACK(mat_stack, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), 
				(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
			TRANSITION_TO(task_ds_zero);
		}
		// All done
		scratch_bak[0] = 1;	
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		transition_to(CUR_TASK);
	}
	if(b == NULL) {
		POP_STACK(mat_stack, 4);
		setup_cleanup(CUR_TASK);
		TRANSITION_TO(task_cleanup);
	}
	uint16_t i = CUR_SCRATCH[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), 
			(uint8_t *)(CUR_SCRATCH + 1), sizeof(uint16_t));
		TRANSITION_TO(task_ds_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}
#endif

void task_d_fc() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	if(CUR_SCRATCH[0] == 0) { // Dense mat mul
		PRINTF("\r\n     Dense MM");
		TASK_REF(task_dm_mul)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, w, (b == NULL) ? dest :  inter, src);
		scratch_bak[0] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_dm_mul);
	} else if(CUR_SCRATCH[0] == 1) { // Bias
		if(b == NULL) {
			POP_STACK(mat_stack, 4);
			setup_cleanup(CUR_TASK);
			TRANSITION_TO(task_cleanup);
		}
		PRINTF("\r\n     Biasing");
		TASK_REF(task_dm_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b, dest, inter);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_dm_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}

void task_s_fc() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->dims, dest->len_dims);
	if(CUR_SCRATCH[0] == 0) { // Sparse mat mul
		PRINTF("\r\n     Sparse MM");
		TASK_REF(task_svm_mul)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, w, (b == NULL) ? dest :  inter, src);
		scratch_bak[0] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_svm_mul);
	} else if(CUR_SCRATCH[0] == 1) { // Bias
		if(b == NULL) {
			POP_STACK(mat_stack, 4);
			setup_cleanup(CUR_TASK);
			TRANSITION_TO(task_cleanup);
		}
		PRINTF("\r\n     Biasing");
		TASK_REF(task_dm_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b, dest, inter);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), 
			(uint8_t *)(CUR_SCRATCH), sizeof(uint16_t));
		TRANSITION_TO(task_dm_add);
	}
	POP_STACK(mat_stack, 4);
	setup_cleanup(CUR_TASK);
	TRANSITION_TO(task_cleanup);
}