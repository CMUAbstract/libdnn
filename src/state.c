#include <libalpaca/alpaca.h>
#include "state.h"
#include "mem.h"

static __fram uint16_t pos_bak;

void push_stack(stack_t *st, mat_t *data[], uint16_t p) {
	for(uint16_t i = 0; i < p; i++) {
		st->data[(st->pos + i) % SAVE_DEPTH] = data[i];
	}
	pos_bak = st->pos + p;
	write_to_gbuf((uint8_t *)&pos_bak, (uint8_t *)&st->pos, sizeof(uint16_t));
}

void pop_stack(stack_t *st, uint16_t p) {
	pos_bak = st->pos - p;
	write_to_gbuf((uint8_t *)&pos_bak, (uint8_t *)&st->pos, sizeof(uint16_t));
}