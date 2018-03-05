#include <string.h>
#include <msp430.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>

#include "nn.h"
#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "fixed.h"
#include "mat.h"
#include "misc.h"

// BLAS Tasks
TASK(TASK_UID_BLAS_OFFSET, task_init_blas);
TASK(TASK_UID_BLAS_OFFSET + 1, task_ds_zero);
TASK(TASK_UID_BLAS_OFFSET + 2, task_ds_add);
TASK(TASK_UID_BLAS_OFFSET + 3, task_dm_add);
TASK(TASK_UID_BLAS_OFFSET + 4, task_dm_mul);
TASK(TASK_UID_BLAS_OFFSET + 5, task_dm_conv);
TASK(TASK_UID_BLAS_OFFSET + 6, task_sm_mul);
TASK(TASK_UID_BLAS_OFFSET + 7, task_sm_conv);

static __hifram fixed data[MAX_LAYER_SIZE];
static __fram mat_t m;
static __fram mat_t *inter;
static __fram mat_t c_filter, c_dest, c_inter, c_src;
static __fram mat_t *c_filter_ptr, *c_dest_ptr, *c_inter_ptr, *c_src_ptr;
static __fram uint scratch_bak[SCRATCH_SIZE];

void task_cleanup_nn();
TASK(TASK_UID_BLAS_OFFSET + 7, task_cleanup_nn);

// Resets a task
static __fram task_t *last_task;
void task_cleanup_nn() {
	PRINTF("\r\n Cleaning up NN");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

void task_init_nn() {
	PRINTF("\r\n Initializing NN");
	inter = &m;
	inter->data = data;
	c_filter_ptr = &c_filter;
	c_dest_ptr = &c_dest;
	c_inter_ptr = &c_inter;
	c_src_ptr = &c_src;
	last_task = CUR_TASK;
	TASK_REF(task_init_blas)->info.return_task = TASK_REF(task_cleanup_nn);
	TRANSITION_TO(task_init_blas);
}

void task_d_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->len_dims, dest->dims);
	uint filters = MAT_GET_DIM(w, 0);
	if(CUR_INFO.scratch[0] == 0) { // Do convolution on all filters
		uint i = CUR_INFO.scratch[1];
		PRINTF("\r\n    Convolving %u", i);
		if(i < filters) {
			TASK_REF(task_dm_conv)->info.return_task = CUR_TASK;
			// Assumes filter, dest, src in that order
			c_inter = MAT_CONSTRAIN(inter, i);
			c_filter = MAT_CONSTRAIN(w, i);
			PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
			TRANSITION_TO(task_dm_conv);
		}
		scratch_bak[0] = 1;	
		scratch_bak[1] = 0;	
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		transition_to(CUR_TASK);
	}
	uint i = CUR_INFO.scratch[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		TRANSITION_TO(task_ds_add);
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 4);
	TRANSITION_TO(task_cleanup_nn);
}


void task_d_conv1d() {
	#if 0
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->len_dims, dest->dims);
	uint filters = MAT_GET_DIM(w, 0);
	if(CUR_INFO.scratch[0] == 0) { // Do convolution on all filters
		uint i = CUR_INFO.scratch[1];
		PRINTF("\r\n    Convolving %u", i);
		if(i < filters) {
			TASK_REF(task_dm_conv)->info.return_task = CUR_TASK;
			// Assumes filter, dest, src in that order
			c_inter = MAT_CONSTRAIN(inter, i);
			c_filter = MAT_CONSTRAIN(w, i);
			c_src_ptr = src;
			// Handles separable filters
			if(stride[0] == 0) {
				c_src= MAT_CONSTRAIN(src, i);
			}
			PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, c_src_ptr);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
			TRANSITION_TO(task_dm_conv1d);
		}
		scratch_bak[0] = 1;	
		scratch_bak[1] = 0;	
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		transition_to(CUR_TASK);
	}
	uint i = CUR_INFO.scratch[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		TRANSITION_TO(task_ds_add);
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 4);
	TRANSITION_TO(task_cleanup_nn);
	#endif
}

void task_s_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->len_dims, dest->dims);
	uint filters = w->sparse.dims[0];
	if(CUR_INFO.scratch[0] == 0) { // Sparse Convolve
		uint i = CUR_INFO.scratch[1];
		uint running_size = CUR_INFO.scratch[2];
		if(i < filters) {
			if(w->sparse.sizes[i] > 0) {
				PRINTF("\r\n     Convolving %u %u %u", i, running_size, w->sparse.sizes[i]);
				TASK_REF(task_sm_conv)->info.return_task = CUR_TASK;
				// Assumes filter, dest, src in that order
				c_filter = MAT_CONSTRAIN(w, running_size);
				c_filter.dims[0] = w->sparse.sizes[i];
				c_inter = MAT_CONSTRAIN(inter, i);
				PUSH_STACK(mat_stack, c_filter_ptr, c_inter_ptr, src);
				scratch_bak[1] = i + 1;
				scratch_bak[2] = running_size + w->sparse.sizes[i];
				write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
				write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
				TRANSITION_TO(task_sm_conv);
			}
			PRINTF("\r\n     Zeroing %u", i);
			TASK_REF(task_ds_zero)->info.return_task = CUR_TASK;
			// Assumes dest, src in that order
			c_inter = MAT_CONSTRAIN(inter, i);
			PUSH_STACK(mat_stack, c_inter_ptr, src);
			scratch_bak[1] = i + 1;
			write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
			TRANSITION_TO(task_ds_zero);
		}
		// All done
		scratch_bak[0] = 1;	
		scratch_bak[1] = 0;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		transition_to(CUR_TASK);
	}
	uint i = CUR_INFO.scratch[1];
	PRINTF("\r\n    Biasing %u", i);
	if(i < filters) {
		TASK_REF(task_ds_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		c_inter = MAT_CONSTRAIN(inter, i);
		c_filter = MAT_CONSTRAIN(b, i);
		c_dest = MAT_CONSTRAIN(dest, i);
		PUSH_STACK(mat_stack, c_filter_ptr, c_dest_ptr, c_inter_ptr);
		scratch_bak[1] = i + 1;
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		TRANSITION_TO(task_ds_add);
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 4);
	TRANSITION_TO(task_cleanup_nn);
}

void task_d_fc() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->len_dims, dest->dims);
	if(CUR_INFO.scratch[0] == 0) { // Dense mat mul
		PRINTF("\r\n     Dense MM");
		TASK_REF(task_dm_mul)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, w, inter, src);
		scratch_bak[0] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		TRANSITION_TO(task_dm_mul);
	} else if(CUR_INFO.scratch[0] == 1) { // Bias
		PRINTF("\r\n     Biasing");
		TASK_REF(task_dm_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b, dest, inter);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		TRANSITION_TO(task_dm_add);
	}
	POP_STACK(mat_stack, 6);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nn);
}

void task_s_fc() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *w= PEEK_STACK(mat_stack, 2);
	mat_t *b = PEEK_STACK(mat_stack, 3);
	mat_reshape(inter, dest->len_dims, dest->dims);
	if(CUR_INFO.scratch[0] == 0) { // Sparse mat mul
		PRINTF("\r\n     Sparse MM");
		TASK_REF(task_sm_mul)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, w, inter, src);
		scratch_bak[0] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		TRANSITION_TO(task_sm_mul);
	} else if(CUR_INFO.scratch[0] == 1) { // Bias
		PRINTF("\r\n     Biasing");
		TASK_REF(task_dm_add)->info.return_task = CUR_TASK;
		// Assumes filter, dest, src in that order
		PUSH_STACK(mat_stack, b, dest, inter);
		scratch_bak[0] = 2;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		TRANSITION_TO(task_dm_add);
	}
	POP_STACK(mat_stack, 4);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_nn);
}
