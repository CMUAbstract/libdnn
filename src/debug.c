#include "debug.h"

#include <msp430.h>
#include <libmspbuiltins/builtins.h>
#include <libmsp/gpio.h>

void pulse(uint16_t length) {
	P8DIR = 0x02;
	P8OUT = 0x02;
	for(uint16_t i = 0; i < length; i++) {
		__delay_cycles(0x400);
	}
	P8OUT = 0x00;
	__delay_cycles(0x400);
}