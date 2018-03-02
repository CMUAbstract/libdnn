#include <string.h>
#include <libio/console.h>
#include <libalpaca/alpaca.h>
#include <libdsp/DSPLib.h>

#include "blas.h"
#include "mem.h"
#include "types.h"
#include "state.h"
#include "fixed.h"
#include "mat.h"

static DSPLIB_DATA(tsrc1, 2) fixed tsrc1[MAX_TILE_SIZE];
static DSPLIB_DATA(tsrc2, 2) fixed tsrc2[MAX_TILE_SIZE];
static DSPLIB_DATA(tdest, 2) fixed tdest[MAX_TILE_SIZE];
static __fram fixed data[MAX_MAT_SIZE];
static __fram fixed data2[MAX_MAT_SIZE];

static __fram mat_t m;
static __fram mat_t *inter;
static __fram mat_t m2;
static __fram mat_t *inter2;

static __fram uint scratch_bak[SCRATCH_SIZE];

static __fram tile_size = 0;

uint greatest_tile_size(uint a, uint max) {
	if(a < max) return a + (a % 2); // Special case when max > a

	uint i = 2;
	uint max_divisor = i;
	while(i < max && i <= a) {
        if(a % i == 0) max_divisor = i;
    	i += 2;
    }
    return max_divisor;
}

void task_cleanup_blas();
void task_sm_mul_addr();
TASK(TASK_UID_BLAS_OFFSET + 8, task_cleanup_blas);
TASK(TASK_UID_BLAS_OFFSET + 9, task_sm_mul_addr);

// Resets a task
static __fram task_t *last_task;
void task_cleanup_blas() {
	// PRINTF("\r\n     Finishing BLAS");
	memset(last_task->info.scratch, 0, sizeof(unsigned int) * SCRATCH_SIZE);
	transition_to(last_task->info.return_task);
}

// Initialize the blas library
void task_init_blas() {
	if(CUR_INFO.scratch[0] == 0) {
		PRINTF("\r\n Initializing BLAS");
		inter = &m;
		inter->data = data;
		inter2 = &m2;
		inter2->data = data2;
		CUR_INFO.scratch[1] = MAX_TILE_SIZE;
		CUR_INFO.scratch[0] = 1;
#ifdef CONFIG_INTERMITTENT
		while(1) {}
#else
		transition_to(CUR_TASK);
#endif
	}

	if(CUR_INFO.scratch[0] == 3) {
		PRINTF("\r\n Waiting to die");
		CUR_INFO.scratch[0] = 1;
#ifdef CONFIG_INTERMITTENT
		while(1) {}
#else
		transition_to(CUR_TASK);
#endif
	} else if(CUR_INFO.scratch[0] == 1 && tile_size == 0){
		PRINTF("\r\n Calibrating; tile size: %u", CUR_INFO.scratch[1]);
		while(true) {
			CUR_INFO.scratch[0] = 2;
			// Might die here so might find a config that is smaller than optimal
			msp_mac_q15_params params;
			params.length = CUR_INFO.scratch[1];
			msp_status status;
			status = msp_mac_q15(&params, tsrc1, tsrc2, tdest);
			msp_checkStatus(status);
			PRINTF("\r\n DONE: %u", status);
			write_to_gbuf((uint8_t *)(CUR_INFO.scratch + 1), (uint8_t *)(&tile_size), sizeof(uint));
			transition_to(CUR_TASK);
		}
	} else if(CUR_INFO.scratch[0] == 2 && tile_size == 0) {
		PRINTF('\r\n Calculating next tile size to try');
		scratch_bak[0] = 3;
		scratch_bak[0] = CUR_INFO.scratch[1] - 5;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		transition_to(CUR_TASK);	
	}
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);

}

// Dense scalar addition
void task_ds_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			fixed w = F_ADD(MAT_GET(src, i, j), MAT_GET(filter, 0));
			MAT_SET(dest, w, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense set mat to all 0s
void task_ds_zero() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		for(uint j = CUR_INFO.scratch[1]; j < cols; j = ++CUR_INFO.scratch[1]) {
			MAT_SET(dest, 0, i, j);
		}
		CUR_INFO.scratch[1] = 0;
	}
	last_task = CUR_TASK;
	POP_STACK(mat_stack, 2);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix addition
void task_dm_add() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(src, 0);
	uint cols = MAT_GET_DIM(src, 1);
	uint common_tile_size = greatest_tile_size(rows * cols, tile_size);
	
	msp_status status;
	msp_add_q15_params params;
	params.length = common_tile_size;
	
	for(uint i = CUR_INFO.scratch[0]; i < rows * cols; i = (CUR_INFO.scratch[0] += common_tile_size)) {
		memcpy(tsrc1, src->data + CUR_INFO.scratch[0], sizeof(fixed) * common_tile_size);
		memcpy(tsrc2, filter->data + CUR_INFO.scratch[0], sizeof(fixed) * common_tile_size);
		status = msp_add_q15(&params, tsrc1, tsrc2, tdest);
		msp_checkStatus(status);
		memcpy(dest->data + CUR_INFO.scratch[0], tdest, sizeof(fixed) * common_tile_size);
	}

	last_task = CUR_TASK;
	POP_STACK(mat_stack, 3);
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix multiplication
void task_dm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);
	uint rows = MAT_GET_DIM(filter, 0);
	uint cols = MAT_GET_DIM(filter, 1);
	uint dcols = MAT_GET_DIM(dest, 1);
	uint common_tile_size = greatest_tile_size(cols, tile_size);
	MAT_RESHAPE(inter, rows, dcols);

	msp_mac_q15_params params;
	params.length = common_tile_size;

	mat_t *inter_tmp = inter;
	if(CUR_INFO.scratch[3] == 0) {
		uint test = cols / common_tile_size;
		if(test % 2 == 0) { // Swapped
			mat_t *tmp = inter_tmp;
			inter_tmp = dest;
			dest = tmp;
		}
	} else if(CUR_INFO.scratch[3] == 2){ // Swapped
		mat_t *tmp = inter_tmp;
		inter_tmp = dest;
		dest = tmp;
	}

	uint j = CUR_INFO.scratch[1];
	for(uint i = CUR_INFO.scratch[0]; i < rows; i = ++CUR_INFO.scratch[0]) {
		memcpy(tsrc1, filter->data + i * cols + j, sizeof(fixed) * common_tile_size);
		for(uint k = CUR_INFO.scratch[2]; k < dcols; k = ++CUR_INFO.scratch[2]) {
			for(uint l = 0; l < common_tile_size; l++) {
				tsrc2[l] = *(src->data + (j + l) * dcols + k); // strided memcpy
			}
			msp_mac_q15(&params, tsrc1, tsrc2, tdest);
#if 0
			fixed w = ((*tdest >> 1) + F_K) >> F_N;
#endif
			if(CUR_INFO.scratch[3] > 0) {
				w = F_ADD(MAT_GET(inter_tmp, i, k), w);
			}
			MAT_SET(dest, w, i, k);
		}
		CUR_INFO.scratch[2] = 0;
	}

	scratch_bak[0] = 0;
	scratch_bak[2] = 0;
	scratch_bak[1] = CUR_INFO.scratch[1] + common_tile_size;
	scratch_bak[3] = (CUR_INFO.scratch[3] == 2) ? 1 : 2;
	if(CUR_INFO.scratch[3] == 0) {
		uint test = cols / common_tile_size;
		scratch_bak[3] = (test % 2 == 0) ? 1 : 2;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	if(!(CUR_INFO.scratch[1] + common_tile_size == cols)) {
		transition_to(CUR_TASK);
	}
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

// Dense matrix convolution
void task_dm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint flayers = MAT_GET_DIM(filter, 0);
	uint frows = MAT_GET_DIM(filter, 1);
	uint fcols = MAT_GET_DIM(filter, 2);
	uint common_tile_size = greatest_tile_size(fcols, tile_size);
	MAT_RESHAPE(inter, rows, cols);

	msp_mac_q15_params params;
	params.length = common_tile_size;

	mat_t *inter_tmp = inter;
	if(CUR_INFO.scratch[4] == 0) {
		uint test = fcols / common_tile_size;
		if(test % 2 == 0) { // Swapped
			mat_t *tmp = inter_tmp;
			inter_tmp = dest;
			dest = tmp;
		}
	} else if(CUR_INFO.scratch[4] == 2){ // Swapped
		mat_t *tmp = inter_tmp;
		inter_tmp = dest;
		dest = tmp;
	}

	memcpy(tsrc1, filter->data + CUR_INFO.scratch[3], sizeof(fixed) * common_tile_size);
	if(common_tile_size > fcols) {
		/* We have to zero an extra element for this case
		Basically, fcols was smaller than max tile size and was odd
		*/
		tsrc1[common_tile_size - 1] = 0;
	}
	uint offset = CUR_INFO.scratch[0] * rows * cols + CUR_INFO.scratch[1] * cols 
		+ CUR_INFO.scratch[2];
	for(uint i = CUR_INFO.scratch[1]; i < rows; i = ++CUR_INFO.scratch[1]) {
		for(uint j = CUR_INFO.scratch[2]; j < cols; j = ++CUR_INFO.scratch[2]) {
			memcpy(tsrc2, (src->data + offset), sizeof(fixed) * common_tile_size);
			msp_mac_q15(&params, tsrc1, tsrc2, tdest);
			fixed w = ((*tdest >> 1) + F_K) >> F_N;
			if(CUR_INFO.scratch[4] > 0) {
				w = F_ADD(MAT_GET(inter_tmp, i, j), w);
			}
			MAT_SET(dest, w, i, j);
			offset++;
		}
		CUR_INFO.scratch[1] = 0;
	}

	scratch_bak[0] = 0;
	scratch_bak[1] = 0;
	scratch_bak[2] = CUR_INFO.scratch[2];
	scratch_bak[3] = CUR_INFO.scratch[3] + common_tile_size;
	if(CUR_INFO.scratch[3] + common_tile_size == frows * fcols) {
		scratch_bak[2] = CUR_INFO.scratch[2] + 1;
		scratch_bak[3] = 0;
	}
	scratch_bak[4] = (CUR_INFO.scratch[4] == 2) ? 1 : 2;
	if(CUR_INFO.scratch[4] == 0) {
		uint test = fcols / common_tile_size;
		scratch_bak[4] = (test % 2 == 0) ? 1 : 2;
	}
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	if(!(CUR_INFO.scratch[3] + common_tile_size == frows * fcols && 
		CUR_INFO.scratch[4] + 1 == flayers)) {
		transition_to(CUR_TASK);
	}

	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}

void task_dm_conv1d() {
	#pragma GCC warning "Not yet implemented"
}

// Sparse matrix multiplication
void task_sm_mul() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0); // n => i
	uint cols = MAT_GET_DIM(src, 0); // m => k
	uint dcols = MAT_GET_DIM(dest, 1); // p => j
	MAT_RESHAPE(inter, rows, dcols);

	uint pos = CUR_INFO.scratch[0];
	uint i = CUR_INFO.scratch[1];
	uint k = CUR_INFO.scratch[2];
	if(pos == 0) {
		k += filter->sparse.offsets[pos];
		pos++;

		scratch_bak[0] = pos;
		scratch_bak[1] = k / cols;
		scratch_bak[2] = k % cols;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		transition_to(CUR_TASK);
	}

	for(uint j = CUR_INFO.scratch[3]; j < dcols; j = ++CUR_INFO.scratch[3]) {
		fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, j));
		w = F_ADD(w, MAT_GET(dest, i, j));
		MAT_SET(inter, w, i, j);
		write_to_gbuf((uint8_t *)(inter->data + dcols * i + j), (uint8_t *)(dest->data + dcols * i + j), sizeof(fixed));
	}
	TRANSITION_TO(task_sm_mul_addr);
}

void task_sm_mul_addr() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint cols = MAT_GET_DIM(src, 0); // m => k
	uint total_elements = MAT_GET_DIM(filter, 0);

	task_t *sm_mul = TASK_REF(task_sm_mul);
	uint orig_pos = sm_mul->info.scratch[0];
	uint pos = orig_pos;
	uint i = sm_mul->info.scratch[1];
	uint k = sm_mul->info.scratch[2];

	k += filter->sparse.offsets[pos];
	pos++;

	scratch_bak[0] = pos;
	scratch_bak[1] = i + k / cols;
	scratch_bak[2] = k % cols;

	sm_mul->info.scratch[3] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(sm_mul->info.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(sm_mul->info.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(sm_mul->info.scratch + 2), sizeof(uint));
	if(orig_pos < total_elements - 1) TRANSITION_TO(task_sm_mul);
	POP_STACK(mat_stack, 3);
	last_task = sm_mul;
	TRANSITION_TO(task_cleanup_blas);
}

#if 0
// Coalesce values of the filter
__fram fixed coalesced_filter[MAX_TILE_SIZE];
void task_sm_conv() {
	/*
		0: idx
		1: pos
		2: still_coalescing?
		3: i
		4: j
		5: destination
		6: coalesce?
	*/
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse_dims[1];
	uint fcols = filter->sparse_dims[2];
	uint common_tile_size = greatest_tile_size(fcols, tile_size);
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter, rows, cols);

	uint idx = CUR_INFO.scratch[0]; // WHAT HAPPENS IF THIS OVERFLOWS?
	uint pos = CUR_INFO.scratch[1];
	if(pos == 0) {
		idx = 0;
		pos = 0;
		char greater = 0;
		while(MAT_GET(filter, pos) == 0) { // Calculate next pos, idx
			greater = 1;
			idx += 255;
			pos++;
		}
		if(greater) idx--; // Fix bug with idx % 255 == 0
		idx += MAT_GET(filter, pos);
		pos++;
		coalesced_filter[idx % common_tile_size] = MAT_GET(filter, pos);
		scratch_bak[0] = idx;
		scratch_bak[1] = pos;
		scratch_bak[2] = idx;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		transition_to(CUR_TASK);
	}

	if(CUR_INFO.scratch[6] == 0) { // Coalesce
		while(CUR_INFO.scratch[2] + common_tile_size > idx) {
			PRINTF("\r\n idx: %u pos: %u tile_size: %u", idx, pos, tile_size);
			char greater = 0;
			while(MAT_GET(filter, pos + 1) == 0) { // Calculate next pos, idx
				greater = 1;
				idx += 255;
				pos++;
			}
			if(greater) idx--; // Fix bug with idx % 255 == 0
			pos++;
			idx += MAT_GET(filter, pos);
			pos++;
			coalesced_filter[idx % common_tile_size] = MAT_GET(filter, pos);
		}
		scratch_bak[0] = idx;
		scratch_bak[1] = pos;
		scratch_bak[2] = idx;
		scratch_bak[6] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 6), (uint8_t *)(CUR_INFO.scratch + 6), sizeof(uint));
		transition_to(CUR_TASK);
	} else if(CUR_INFO.scratch[6] == 1) { // Apply filter to matrix
		idx -= common_tile_size;
		uint k = idx / (fcols * frows); // Layers
		uint l = (idx % (fcols * frows)); // Rows
		uint n = idx % fcols; // Cols
		PRINTF("\r\n idx: %u pos: %u k: %u l: %u n: %u total_elements: %u", idx, pos, k, l, n, total_elements);
		msp_mac_q15_params params;
		params.length = common_tile_size;

		if(CUR_INFO.scratch[5] == 2) {
			mat_t *tmp = inter;
			inter = dest;
			dest = tmp;
			PRINTF("\r\n SWAPPED=> Inter: dest Dest: inter");
		} else if(CUR_INFO.scratch[5] == 1) {
			PRINTF("\r\n Inter: inter Dest: dest");
		}

		memcpy(tsrc1, coalesced_filter, sizeof(fixed) * common_tile_size);
		uint offset = k * rows * cols + l * cols + n;
		for(uint i = CUR_INFO.scratch[3]; i < rows; i = ++CUR_INFO.scratch[3]) {
			for(uint j = CUR_INFO.scratch[4]; j < cols; j = ++CUR_INFO.scratch[4]) {
				memcpy(tsrc2, (src->data + offset), sizeof(fixed) * common_tile_size);
				msp_mac_q15(&params, tsrc1, tsrc2, tdest);
				fixed w = *tdest;
				PRINTF("\r\nfilter: %i src: %i w: %u", coalesced_filter[0], *(src->data + offset), w);
				if(CUR_INFO.scratch[5] > 0) {
					w = F_ADD(MAT_GET(inter, i, j), w);
				}
				MAT_SET(dest, w, i, j);
				offset++;
			}
			CUR_INFO.scratch[4] = 0;
		}

		scratch_bak[3] = 0;
		scratch_bak[4] = 0;
		scratch_bak[6] = 0;
		scratch_bak[5] = (CUR_INFO.scratch[5] == 2) ? 1 : 2;
		if(CUR_INFO.scratch[5] == 0) {
			scratch_bak[5] = 2;
		}
		memset(coalesced_filter, 0, sizeof(fixed) * MAX_TILE_SIZE);
		write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
		if(CUR_INFO.scratch[5] == 2 && n + common_tile_size == fcols && l + 1 == frows){
			scratch_bak[6] = 2;
			write_to_gbuf((uint8_t *)(scratch_bak + 6), (uint8_t *)(CUR_INFO.scratch + 6), sizeof(uint));
			transition_to(CUR_TASK);
		} else if(CUR_INFO.scratch[1] < total_elements - 1) {
			write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
			write_to_gbuf((uint8_t *)(scratch_bak + 6), (uint8_t *)(CUR_INFO.scratch + 6), sizeof(uint));
			transition_to(CUR_TASK);
		}
		POP_STACK(mat_stack, 3);
		last_task = CUR_TASK;
		TRANSITION_TO(task_cleanup_blas);
	} else if(CUR_INFO.scratch[6] == 2) {
		if(CUR_INFO.scratch[5] == 2) {
			for(uint i = CUR_INFO.scratch[3]; i < rows; i = ++CUR_INFO.scratch[3]) {
				for(uint j = CUR_INFO.scratch[4]; j < cols; j = ++CUR_INFO.scratch[4]) {
					MAT_SET(dest, MAT_GET(inter, i, j), i, j);
				}
			}
		}
		if(CUR_INFO.scratch[1] < total_elements - 1) {
			scratch_bak[6] = 0;
			write_to_gbuf((uint8_t *)(scratch_bak + 6), (uint8_t *)(CUR_INFO.scratch + 6), sizeof(uint));
			transition_to(CUR_TASK);
		}
		POP_STACK(mat_stack, 3);
		last_task = CUR_TASK;
		TRANSITION_TO(task_cleanup_blas);
	}
}
#else
void task_sm_conv() {
	mat_t *src = PEEK_STACK(mat_stack, 0);
	mat_t *dest = PEEK_STACK(mat_stack, 1);
	mat_t *filter = PEEK_STACK(mat_stack, 2);

	uint rows = MAT_GET_DIM(dest, 0);
	uint cols = MAT_GET_DIM(dest, 1);

	uint frows = filter->sparse_dims[1];
	uint fcols = filter->sparse_dims[2];
	uint total_elements = MAT_GET_DIM(filter, 0);
	MAT_RESHAPE(inter, rows, cols);
	MAT_RESHAPE(inter2, rows, cols);

	uint idx = CUR_INFO.scratch[0]; // WHAT HAPPENS IF THIS OVERFLOWS?
	uint pos = CUR_INFO.scratch[1];
	uint k = idx / (fcols * frows); // Layers
	uint l = (idx % (fcols * frows)) / fcols; // Rows
	uint n = idx % fcols; // Cols
	if(pos == 0) {
		idx += filter->sparse.offsets[pos];
		pos++;
		scratch_bak[0] = idx;
		scratch_bak[1] = pos;
		scratch_bak[5] = 1;
		write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
		write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
		transition_to(CUR_TASK);
	}

	// // PRINTF("\r\n Convolving cols: %u rows: %u idx:  %u pos:  %u k:  %u l:  %u n:  %u tileX:  %u tileY: %u val: %i", 
	// 	rows, cols, idx, pos, k, l, n, 8, 8, MAT_GET(filter, pos));

	mat_t *prev_dest = (CUR_INFO.scratch[4] % 2 == 0) ? inter2 : inter;
	if(pos < total_elements - 1) {
		dest = (CUR_INFO.scratch[4] % 2 == 0) ? inter : inter2; // swapped
	}

	if(CUR_INFO.scratch[5] == 0) {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				w = F_ADD(w, MAT_GET(prev_dest, i, j));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[3] = 0;
		}
	} else {
		for(uint i = CUR_INFO.scratch[2]; i < rows; i = ++CUR_INFO.scratch[2]) {
			for(uint j = CUR_INFO.scratch[3]; j < cols; j = ++CUR_INFO.scratch[3]) {
				fixed w = F_MUL(MAT_GET(filter, pos), MAT_GET(src, k, i + l, j + n));
				MAT_SET(dest, w, i, j);
			}
			CUR_INFO.scratch[3] = 0;
		}
	}

	idx += filter->sparse.offsets[pos];
	pos++;
	
	scratch_bak[0] = idx;
	scratch_bak[1] = pos;
	scratch_bak[2] = 0;
	scratch_bak[3] = 0;
	scratch_bak[4] = ~CUR_INFO.scratch[4];
	scratch_bak[5] = 0;
	write_to_gbuf((uint8_t *)(scratch_bak), (uint8_t *)(CUR_INFO.scratch), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 1), (uint8_t *)(CUR_INFO.scratch + 1), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 2), (uint8_t *)(CUR_INFO.scratch + 2), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 3), (uint8_t *)(CUR_INFO.scratch + 3), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 4), (uint8_t *)(CUR_INFO.scratch + 4), sizeof(uint));
	write_to_gbuf((uint8_t *)(scratch_bak + 5), (uint8_t *)(CUR_INFO.scratch + 5), sizeof(uint));
	if(CUR_INFO.scratch[1] < total_elements - 1) transition_to(CUR_TASK);
	POP_STACK(mat_stack, 3);
	last_task = CUR_TASK;
	TRANSITION_TO(task_cleanup_blas);
}
#endif