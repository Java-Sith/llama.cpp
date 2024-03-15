#ifndef GGML_XRT_IMPLEMENTATION
#define GGML_XRT_IMPLEMENTATION

#include "ggml.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <iostream>

void mul_mat(float* src0, float* src1, float *dst, int m, int n, int k);

#endif 
