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

float** naiveMatrixMultiply(float** matrix1, int rows1, int cols1, float** matrix2, int cols2);

#endif 
