#ifndef LEA_H
#define LEA_H

#include <stdint.h>
#include <libmspdriver/driverlib.h>
#include <libfixed/fixed.h>

extern fixed tsrc1[CONFIG_TILE_SIZE];
extern fixed tsrc2[CONFIG_TILE_SIZE];
extern fixed tsrc3[CONFIG_TILE_SIZE];
extern fixed tdest1[CONFIG_TILE_SIZE];
extern fixed tdest2[CONFIG_TILE_SIZE];

extern DMA_initParam dma_config;

uint16_t check_calibrate(void);
uint16_t greatest_tile_size(uint16_t dim, uint16_t max);
void DMA_startSleepTransfer(uint16_t channel);

#endif