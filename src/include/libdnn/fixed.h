#ifndef FIXED_H
#define FIXED_H

#include "types.h"

#define F_N 5
#define F_ONE (1 << F_N)
#define F_K (1 << (F_N - 1))
#define F_MASK (~(F_ONE | (F_ONE - 1)))

#define F_LIT(f) (fixed)(f * F_ONE)
#define F_TO_FLOAT(f) (float)(f) / F_ONE 
#define F_ADD(a, b) a + b
#define F_SUB(a, b) a - b
#define F_MUL(a, b) f_mul(a, b)
#define F_DIV(a, b) f_div(a, b)
#define F_ROUND(a) f_round(a)
#define F_SQRT(a) f_sqrt(a)
#define F_SIN(a) f_sin(a)
#define F_COS(a) f_cos(a)
#define F_LT(a, b) a < b
#define F_LTE(a, b) a <= b
#define F_GT(a, b) a > b
#define F_GTE(a, b) a >= b
#define F_EQ(a, b) a == b
#define F_NEQ(a, b) a != b

// Comment out middle two lines for int arithmetic to work
static inline fixed f_mul(fixed a, fixed b) {
    signed int tmp = a * b;
    tmp += F_K;
    tmp >>= F_N;
    return (fixed)tmp;
};

static inline fixed f_div(fixed a, fixed b) {
    signed long tmp = a << F_N;
    if((tmp >= 0 && b >= 0) || (tmp < 0 && b < 0)) {
    	tmp += b / 2;
    } else {
    	tmp -= b / 2;
    }
    return (fixed)(tmp / b);
}

static inline fixed f_round(fixed a) {
	return a & (F_MASK);
}

static inline fixed f_sqrt(fixed a) {
	fixed tmp = F_MUL(F_LIT(0.5), a);
#ifdef CONFIG_FIXED_PRECISE
	for(uint i = 0; i < 8; i++) {
#else
	for(uint i = 0; i < 4; i++) {
#endif
		tmp = F_MUL(F_LIT(0.5), F_ADD(tmp, F_DIV(a, tmp)));
	}
	return tmp;
}

const fixed PI = F_LIT(3.1415926);
const fixed TWO_PI = F_LIT(6.2831853);
const fixed HALF_PI = F_LIT(1.5707963);
static inline fixed f_cos(fixed a) {
	fixed tmp = a;
	// Scale
	if(F_LT(TWO_PI, tmp)) {
		fixed close = F_MUL(TWO_PI, -F_ROUND(F_DIV(tmp, TWO_PI)));
		tmp = F_ADD(tmp, close);
	} else if(F_LT(tmp, -TWO_PI)) {
		fixed close = F_MUL(TWO_PI, F_ROUND(F_DIV(-tmp, TWO_PI)));
		tmp = F_ADD(tmp, close);
	}

	// Center around -PI and PI
	if (F_LT(tmp, -PI)) {
    	tmp = F_ADD(TWO_PI, tmp);
	} else if (F_LT(PI, tmp)) {
	    tmp = F_ADD(-TWO_PI, tmp);
	}

	// Shift
	tmp = F_ADD(tmp, HALF_PI);
	if (F_LT(PI, tmp)) {
		tmp = F_ADD(-TWO_PI, tmp);
	}

	// Apply cos/sin
	fixed first_term = F_MUL(F_LIT(1.27323954), tmp);
	fixed second_term = F_MUL(F_MUL(F_LIT(0.405284735), tmp), tmp);
	if(F_LT(tmp, F_LIT(0))){
	    tmp = first_term + second_term;
#ifdef CONFIG_FIXED_PRECISE
	    if(F_LT(tmp, F_LIT(0))) {
	    	return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, -tmp), -tmp)), tmp);
	    }
	    return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, tmp), -tmp)), tmp);
#else
	    return tmp;
#endif
	}
	tmp = first_term - second_term;
#ifdef CONFIG_FIXED_PRECISE
	if(F_LT(tmp, F_LIT(0))) {
    	return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, -tmp), -tmp)), tmp);
    }
    return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, tmp), -tmp)), tmp);
#else
    return tmp;
#endif
}

static inline fixed f_sin(fixed a) {
	fixed tmp = a;
	// Scale
	if(F_LT(TWO_PI, tmp)) {
		fixed close = F_MUL(TWO_PI, -F_ROUND(F_DIV(tmp, TWO_PI)));
		tmp = F_ADD(tmp, close);
	} else if(F_LT(tmp, -TWO_PI)) {
		fixed close = F_MUL(TWO_PI, F_ROUND(F_DIV(-tmp, TWO_PI)));
		tmp = F_ADD(tmp, close);
	}

	// Center around -PI and PI
	if (F_LT(tmp, -PI)) {
    	tmp = F_ADD(TWO_PI, tmp);
	} else if (F_LT(PI, tmp)) {
	    tmp = F_ADD(-TWO_PI, tmp);
	}

	// Apply cos/sin
	fixed first_term = F_MUL(F_LIT(1.27323954), tmp);
	fixed second_term = F_MUL(F_MUL(F_LIT(0.405284735), tmp), tmp);
	if(F_LT(tmp, F_LIT(0))){
	    tmp = first_term + second_term;
#ifdef CONFIG_FIXED_PRECISE
	    if(F_LT(tmp, F_LIT(0))) {
	    	return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, -tmp), -tmp)), tmp);
	    }
	    return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, tmp), -tmp)), tmp);
#else
	    return tmp;
#endif
	}
	tmp = first_term - second_term;
#ifdef CONFIG_FIXED_PRECISE
	if(F_LT(tmp, F_LIT(0))) {
    	return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, -tmp), -tmp)), tmp);
    }
    return F_ADD(F_MUL(F_LIT(0.225), F_ADD(F_MUL(tmp, tmp), -tmp)), tmp);
#else
    return tmp;
#endif
}

#endif