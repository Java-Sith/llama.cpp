/*
 * Copyright 2022-2024
 * Author: Luis D. Prieto-Sibaja <prieto.luisdaniel@gmail.com>
 */

#ifndef __DEQUANTIZE4_H__
#define __DEQUANTIZE4_H__

#define AP_INT_MAX_W 32768

#include "common/config.h"

#ifndef M_COLS
static constexpr int kCols = AP_INT_MAX_W; // Default columns size
#else
static constexpr int kCols = M_COLS;
#endif

#ifndef M_ROWS
static constexpr int kRows = AP_INT_MAX_W; // Default rows size
#else
static constexpr int kRows = M_ROWS;
#endif

// The number of elements processed in each block (QK4)
#ifndef QK_K
static constexpr int QK_K = 256;
#endif

typedef struct {
    ap_fixed<16, 5> d;    // half-precision scale
    ap_fixed<16, 5> dmin; // half-precision min
    ap_uint<8> scales[K_SCALE_SIZE]; // 6-bit quantized scales and mins
    ap_uint<8> qs[QK_K / 2];         // 4-bit quantized values
} block_q4_K;

extern "C" {
// Dequantize function, based on Q4 block tensor layout
void dequantize4(block_q4_K *x, float *y, uint64_t k);
}

#endif // __DEQUANTIZE_H__
