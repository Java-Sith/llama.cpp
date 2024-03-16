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

float** allocateMatrix(int rows, int cols);
void deallocateMatrix(float** matrix, int rows);
void initializeMatrix(float** matrix, int rows, int cols);
float** matrixMultiplication(float** mat1, float** mat2, int rows1, int cols1, int cols2);
void printMatrix(float** matrix, int rows, int cols);

#endif 
