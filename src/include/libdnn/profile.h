#ifndef PROFILE_H
#define PROFILE_H

#include <libfixed/fixed.h>

#ifndef CONFIG_PROFILE
#pragma GCC warning "no profiling"
	#define inc_ld(v) (void)0
	#define inc_ld_vec(v) (void)0
	#define inc_st(v) (void)0
	#define inc_st_vec(v) (void)0
	#define inc_add(v) (void)0
	#define inc_mul(v) (void)0
	#define inc_addr_add(v) (void)0
	#define inc_addr_mul(v) (void)0
	#define inc_add_vec(v) (void)0
	#define inc_fir_vec(v, f) (void)0
	#define inc_mac_vec(v) (void)0
	#define print_stats() (void)0
#else
void inc_ld(uint);
void inc_st(uint);
void inc_add(uint);
void inc_mul(uint);
void inc_addr_add(uint);
void inc_addr_mul(uint);

	#ifdef CONFIG_LEA
		void inc_ld_vec(uint);
		void inc_st_vec(uint);
		void inc_mac_vec(uint);
		void inc_fir_vec(uint, uint);
		void inc_add_vec(uint);
	#endif

void print_stats();
#endif

#endif