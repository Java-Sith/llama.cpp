#pragma once

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

#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    ggml_fp16_t d;          // delta
    ggml_fp16_t m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_fp16_t) + QK4_1 / 2, "wrong q4_1 block size/padding");

typedef void (*ggml_xrt_func_t)(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void quantize_row_q4_0(const float * x, block_q4_0 * y, int k);
void quantize_row_q4_0_reference(const float * x, block_q4_0 * y, int k);
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int k);
float** allocateMatrix(int rows, int cols);
void deallocateMatrix(float** matrix, int rows);
void initializeMatrix(float** matrix, int rows, int cols);
float** matrixMultiplication(float** mat1, float** mat2, int rows1, int cols1, int cols2);
void printMatrix(float** matrix, int rows, int cols);
GGML_API bool ggml_xrt_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);