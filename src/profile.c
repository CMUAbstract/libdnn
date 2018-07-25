#include "profile.h"

#include <msp430.h>
#include <string.h>
#include <libmspbuiltins/builtins.h>
#include <libio/console.h>
#include <libfixed/fixed.h>

#include "mem.h"
#include "misc.h"

#ifdef CONFIG_PROFILE
void prof_pulse(uint16_t length) {
	P8DIR = 0x02;
	P8OUT = 0x02;
	for(uint16_t i = 0; i < length; i++) {
		__delay_cycles(0x10);
	}
	P8OUT = 0x00;
}
#endif

#if CONFIG_PROFILE == 1
#define stats_lenGTH 0x10
#define STAT_NAME_LENGTH 0x10

typedef struct {
	char name[STAT_NAME_LENGTH];
	uint32_t invocs;
	uint32_t ops;
} stat_t;

typedef struct {
	char name[STAT_NAME_LENGTH];
	stat_t stats[stats_lenGTH];
	uint16_t stats_len;
} section_t;

typedef struct {
	section_t overall;
	section_t sections[stats_lenGTH / 2];
	uint16_t sections_len;
} prof_t;

static __fram prof_t profiler;

void prof_inc(char *name, uint16_t invocs, uint16_t ops) {
	uint16_t stats_len = profiler.overall.stats_len;
	for(uint16_t i = 0; i < stats_len; i++) {
		if(strcmp(name, profiler.overall.stats[i].name) == 0) {
			profiler.overall.stats[i].invocs += invocs;
			profiler.overall.stats[i].ops += ops;
			return;
		}
	}
	strcpy(profiler.overall.stats[stats_len].name, name);
	profiler.overall.stats[stats_len].invocs = invocs;
	profiler.overall.stats[stats_len].ops = ops;
	profiler.overall.stats_len++;
}

void prof_print() {
	PRINTF("\r\n=========Stats=========");
	PRINTF("\r\n[{");
	PRINTF("\r\n\"backend\": \"\",");
	PRINTF("\r\n\"target\": \"\",");
	PRINTF("\r\n\"overall\": {");
	for(uint16_t i = 0; i < profiler.overall.stats_len; i++) {
		PRINTF("\r\n  \"%s\": {\"invocs\": %n, \"ops\": %n}",
			profiler.overall.stats[i].name, 
			profiler.overall.stats[i].invocs, 
			profiler.overall.stats[i].ops);
		if(i != profiler.overall.stats_len - 1) PRINTF(",");
	}
	PRINTF("\r\n},");
	PRINTF("\r\n\"sections\": {");
	for(uint16_t i = 0; i < profiler.sections_len; i++) {
		PRINTF("\r\n \"%s\": {", profiler.sections[i].name);
		for(uint16_t j = 0; j < profiler.sections[i].stats_len; j++) {
			PRINTF("\r\n  \"%s\": {\"invocs\": %n, \"ops\": %n}",
				profiler.sections[i].stats[j].name, 
				profiler.sections[i].stats[j].invocs, 
				profiler.sections[i].stats[j].ops);
			if(j != profiler.sections[i].stats_len - 1) PRINTF(",");
		}
		PRINTF("\r\n }");
		if(i != profiler.sections_len - 1) PRINTF(",");
	}
	PRINTF("\r\n}");
	PRINTF("\r\n}]");
	PRINTF("\r\n=======================");
}

void prof(prof_control_t t, char *layer) {
	switch(t) {
		case SECTION:
			if(profiler.sections_len == 0) {
				strcpy(profiler.sections[0].name, layer);
				profiler.sections[0].stats_len = profiler.overall.stats_len;
				for(uint16_t i = 0; i < profiler.overall.stats_len; i++) {
					strcpy(profiler.sections[0].stats[i].name, 
						profiler.overall.stats[i].name);
					profiler.sections[0].stats[i].invocs = 
						profiler.overall.stats[i].invocs;
					profiler.sections[0].stats[i].ops = 
						profiler.overall.stats[i].ops;
				}
				profiler.sections_len++;
			} else {
				strcpy(profiler.sections[profiler.sections_len].name, layer);
				uint16_t stats_len = 
					profiler.sections[profiler.sections_len - 1].stats_len;
				profiler.sections[profiler.sections_len].stats_len = stats_len;
				for(uint16_t i = 0; i < profiler.overall.stats_len; i++) {
					if(i > profiler.sections[profiler.sections_len - 1].stats_len) {
						strcpy(profiler.sections[profiler.sections_len].stats[i].name,
							profiler.overall.stats[i].name);
						profiler.sections[profiler.sections_len].stats[i].invocs = 
							profiler.overall.stats[i].invocs;
						profiler.sections[profiler.sections_len].stats[i].ops = 
							profiler.overall.stats[i].ops;
						continue;
					}
					strcpy(profiler.sections[profiler.sections_len].stats[i].name,
						profiler.sections[profiler.sections_len - 1].stats[i].name);
					profiler.sections[profiler.sections_len].stats[i].invocs = 
						profiler.overall.stats[i].invocs - 
						profiler.sections[profiler.sections_len - 1].stats[i].invocs;
					profiler.sections[profiler.sections_len].stats[i].ops = 
						profiler.overall.stats[i].ops - 
						profiler.sections[profiler.sections_len - 1].stats[i].ops;
				}
				profiler.sections_len++;
			}
		default: break;
	}
}

#endif
