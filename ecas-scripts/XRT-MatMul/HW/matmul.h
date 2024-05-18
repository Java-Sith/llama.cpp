#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

// FIXME: Modify to adjust
static constexpr int kARows = 2;
#ifndef B_COLS
static constexpr int kBCols = 32768;
#else
static constexpr int kBCols = B_COLS;
#endif
#ifndef C_COLS
static constexpr int kCCols = 32768;
#else
static constexpr int kCCols = C_COLS;
#endif
#ifndef BUS
static constexpr int kBusWidth = 2048; //2048
#else
static constexpr int kBusWidth = BUS;
#endif
//#define USE_FLOAT8

#ifdef USE_FLOAT32
static constexpr int kDataWidth = 32;
#elif defined(USE_FLOAT16)
static constexpr int kDataWidth = 16;
#elif defined(USE_FLOAT8)
static constexpr int kDataWidth = 8;
#elif defined(USE_FLOAT4)
static constexpr int kDataWidth = 4;
#elif defined(USE_FIXED16)
static constexpr int kFxPDataWidth = 16;
static constexpr int kFxPDataInt = 6;
static constexpr int kDataWidth = kFxPDataWidth;
using DataT = ap_fixed<kFxPDataWidth, kFxPDataInt>;
#else
static constexpr int kFxPDataWidth = 8;
static constexpr int kFxPDataInt = 4;
static constexpr int kDataWidth = kFxPDataWidth;
using DataT = ap_fixed<kFxPDataWidth, kFxPDataInt>;
#endif

static constexpr int kPackets = kBusWidth / kDataWidth;

using RawDataT = ap_uint<kBusWidth>;
using StreamT = hls::stream<RawDataT>;

extern "C" {
void matmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols);
}

#endif // __MATMUL_H__
