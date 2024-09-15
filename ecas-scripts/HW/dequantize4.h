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
static constexpr int QK_K = 32;
#endif

extern "C" {
// Dequantize function, based on Q4 block tensor layout
void dequantize4(uint8_t *in, RawDataT *out, uint64_t size);
}

#endif // __DEQUANTIZE_H__
