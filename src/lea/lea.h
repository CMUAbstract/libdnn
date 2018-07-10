#ifndef LEA_H
#define LEA_H

#include <stdint.h>

uint16_t check_calibrate(void);
void greatest_tile_size(uint16_t dim, uint16_t max);
void DMA_startSleepTransfer(uint16_t channel);

#endif