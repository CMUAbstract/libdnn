#ifndef PROFILE_H
#define PROFILE_H

#include <stdint.h>
#include <libfixed/fixed.h>

#if CONFIG_PROFILE == 1
	typedef enum {
		OPEN,
		SECTION,
		CLOSE,
	} prof_control_t;

	void prof_pulse(uint16_t length);
	void prof_inc(char *name, uint16_t invoc, uint16_t ops);
	void prof_print();
	void prof(prof_control_t t, char *layer);
#elif CONFIG_PROFILE == 2
#pragma message "pulse only"
	void prof_pulse(uint16_t length);
	#define prof_inc(n, i, o) (void)0
	#define prof_print(t) (void)0
	#define prof(t, l) (void)0
#else
#pragma message "no profiling"
	#define prof_pulse(l) (void)0
	#define prof_inc(n, i, o) (void)0
	#define prof_print(t) (void)0
	#define prof(t, l) (void)0
#endif

#endif