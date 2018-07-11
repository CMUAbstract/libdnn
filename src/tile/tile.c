#include "tile.h"

uint16_t greatest_tile_size(uint16_t a, uint16_t max) {
	uint16_t i = 1;
	uint16_t max_divisor = i;
	while(i < max && i <= a) {
        if(a % i == 0) max_divisor = i;
        i++;
    }
    return max_divisor;
}

uint16_t greatest_common_tile_size(uint16_t a, uint16_t b, uint16_t max) {
	uint16_t i = 1;
	uint16_t max_divisor = i;
	while(i < max && i <= a) {
        if(a % i == 0 && b % i == 0) max_divisor = i;
        i++;
    }
    return max_divisor;
}