#ifndef FIXED_H
#define FIXED_H

#include "types.h"

#define F_N 5
#define F_ONE (1 << F_N)
#define F_K (1 << (F_N - 1))

#define F_LIT(f) (fixed)(f * F_ONE)
#define F_TO_FLOAT(f) (float)(f) / F_ONE 
#define F_ADD(a, b) a + b
#define F_MUL(a, b) f_mul(a, b)
#define F_LT(a, b) a < b

// Commented out for the sake of int arithmetic to work
static inline fixed f_mul(fixed a, fixed b) {
    signed int tmp = a * b;
    // tmp += F_K;
    // tmp >>= F_N;
    return (fixed)tmp;
};

#endif