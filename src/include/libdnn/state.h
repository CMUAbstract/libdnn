#ifndef STATE_H
#define STATE_H
#include <libmat/mat.h>

#define SAVE_DEPTH 0x10

typedef struct {
	mat_t *data[SAVE_DEPTH];
	uint16_t pos;
} stack_t;

extern stack_t *mat_stack;

#define STACK_NUMARGS(...)  (sizeof((mat_t*[]){__VA_ARGS__})/sizeof(mat_t*))

#define PUSH_STACK(st, ...) push_stack(st, (mat_t*[]){__VA_ARGS__}, STACK_NUMARGS(__VA_ARGS__))

#define POP_STACK(st, p) pop_stack(st, p)

#define PEEK_STACK(st, p) st->data[(st->pos - p - 1) % SAVE_DEPTH]

void push_stack(stack_t *, mat_t *[], uint16_t);
void pop_stack(stack_t *, uint16_t);

#endif