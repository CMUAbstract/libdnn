#ifndef PROFILE_H
#define PROFILE_H

#include <libfixed/fixed.h>

#ifndef CONFIG_PROFILE
#pragma message "no profiling"
	#define prof_inc(n, i, o) (void)0
	#define prof_print() (void)0
#else
	void prof_inc(char *name, uint16_t invoc, uint16_t ops);
	void prof_print();
#endif

#endif