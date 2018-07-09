#ifndef CLEANUP_H
#define CLEANUP_H

#include <libalpaca/alpaca.h>

void setup_cleanup(task_t *);
void task_cleanup();
extern TASK_DEC(task_cleanup);

#endif