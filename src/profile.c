#include "profile.h"

#include <string.h>
#include <libio/console.h>
#include <libfixed/fixed.h>

#include "mem.h"
#include "misc.h"

#ifdef CONFIG_PROFILE
#define STAT_LENGTH 0x10
#define STAT_NAME_LENGTH 0x10

typedef struct {
	char name[STAT_NAME_LENGTH];
	uint32_t invoc;
	uint32_t ops;
} stat;

static __fram stat stats[0x10];
static __fram uint16_t stat_len = 0;

void prof_inc(char *name, uint16_t invoc, uint16_t ops) {
	ld_invoc += v;
	for(uint16_t i = 0; i < stat_len; i++) {
		if(strcmp(name, stats[i].name) == 0) {
			stats[stat_len].invoc += invoc;
			stats[stat_len].ops += ops;
			return;
		}
	}
	strcpy(stats[stat_len].name, name);
	stats[stat_len].invoc = invoc;
	stats[stat_len].ops = ops;
	stat_len++;
}

void prof_print() {
	PRINTF("\r\n=========Stats=========");
	PRINTF("\r\n [");
	for(uint16_t i = 0; i < stat_len; i++) {
		PRINTF("\r\n {\"%s\": {\"invoc\": %n, \"ops\": %n}}",
			stats[i].name, stats[i].invoc, stats[i].ops);
		if(i != stat_len - 1) PRINTF(",");
	}
	PRINTF("\r\n ]");
	PRINTF("\r\n=======================");
}

#endif
