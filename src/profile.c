#include <libio/console.h>
#include <libfixed/fixed.h>
#include "profile.h"
#include "mem.h"

#ifdef CONFIG_PROFILE
typedef struct {
	uint16_t count;
	unsigned long size;
} vec_count;

static __fram unsigned long ld_count = 0;
static __fram unsigned long st_count = 0;
static __fram vec_count ld_vec_count = {.count = 0, .size = 0.};
static __fram vec_count st_vec_count = {.count = 0, .size = 0.};
static __fram unsigned long add_count = 0;
static __fram unsigned long mul_count = 0;
static __fram unsigned long addr_add_count = 0;
static __fram unsigned long addr_mul_count = 0;
static __fram vec_count add_vec_count = {.count = 0, .size = 0.};
static __fram vec_count fir_vec_count = {.count = 0, .size = 0.};
static __fram vec_count mac_vec_count = {.count = 0, .size = 0.};

void inc_ld(uint16_t v) {
	ld_count += v;
}

void inc_st(uint16_t v) {
	st_count += v;
}

void inc_ld_vec(uint16_t v) {
	ld_vec_count.size += v;
	ld_vec_count.count++;
}

void inc_st_vec(uint16_t v) {
	st_vec_count.size += v;
	st_vec_count.count++;
}

void inc_add(uint16_t v) {
	add_count += v;
}

void inc_mul(uint16_t v) {
	mul_count += v;
}

void inc_addr_add(uint16_t v) {
	addr_add_count += v;
}

void inc_addr_mul(uint16_t v) {
	addr_mul_count += v;
}

void inc_mac_vec(uint16_t v) {
	mac_vec_count.size += v;
	mac_vec_count.count++;
}

void inc_fir_vec(uint16_t v, uint16_t f) {
	fir_vec_count.size += v;
	fir_vec_count.count++;
}

void inc_add_vec(uint16_t v) {
	add_vec_count.size += v;
	add_vec_count.count++;
}

void print_stats() {
#if CONFIG_CONSOLE
	PRINTF("\r\n=========Stats=========");
	PRINTF("\r\n{\"ld_count\": %n,", ld_count);
	PRINTF("\r\n \"st_count\": %n,", st_count);
	PRINTF("\r\n \"add_count\": %n,", add_count);
	PRINTF("\r\n \"mul_count\": %n,", mul_count);
	PRINTF("\r\n \"addr_add_count\": %n,", addr_add_count);
	#ifdef CONFIG_LEA
		PRINTF("\r\n \"addr_mul_count\": %n,", addr_mul_count);
		PRINTF("\r\n \"add_vec_count\": {\"count\": %u, \"size\": %n},", 
			add_vec_count.count, add_vec_count.size);
		PRINTF("\r\n \"fir_vec_count\": {\"count\": %u, \"size\": %n},", 
			fir_vec_count.count, fir_vec_count.size);
		PRINTF("\r\n \"mac_vec_count\": {\"count\": %u, \"size\": %n},", 
			mac_vec_count.count, mac_vec_count.size);
		PRINTF("\r\n \"ld_vec_count\": {\"count\": %u, \"size\": %n},", 
			ld_vec_count.count, ld_vec_count.size);
		PRINTF("\r\n \"st_vec_count\": {\"count\": %u, \"size\": %n}}", 
			st_vec_count.count, st_vec_count.size);
	#else
		PRINTF("\r\n \"addr_mul_count\": %n}", addr_mul_count);
	#endif
		PRINTF("\r\n=======================");
#endif
}

#endif
