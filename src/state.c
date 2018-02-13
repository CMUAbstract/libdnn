#include <libalpaca/alpaca.h>
#include "state.h"
#include "mem.h"

static __fram uint pos_bak;

void push_stack(stack_t *st, mat_t *data[], uint p) {
	for(uint i = 0; i < p; i++) {
		st->data[(st->pos + i) % SAVE_DEPTH] = data[i];
	}
	pos_bak = st->pos + p;
	write_to_gbuf((uint8_t *)&pos_bak, (uint8_t *)&st->pos, sizeof(uint));
}

void pop_stack(stack_t *st, uint p) {
	pos_bak = st->pos - p;
	write_to_gbuf((uint8_t *)&pos_bak, (uint8_t *)&st->pos, sizeof(uint));
}