#ifndef PROFILE_H
#define PROFILE_H

#include <libfixed/fixed.h>

#ifndef CONFIG_PROFILE
#pragma message "no profiling"
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
void inc_ld(uint16_t);
void inc_st(uint16_t);
void inc_add(uint16_t);
void inc_mul(uint16_t);
void inc_addr_add(uint16_t);
void inc_addr_mul(uint16_t);

	#ifdef CONFIG_LEA
		void inc_ld_vec(uint16_t);
		void inc_st_vec(uint16_t);
		void inc_mac_vec(uint16_t);
		void inc_fir_vec(uint16_t, uint16_t);
		void inc_add_vec(uint16_t);
	#endif

void print_stats();
#endif

#endif