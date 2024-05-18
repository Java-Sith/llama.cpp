#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

static constexpr int kBusWidth = 512;
static constexpr int kDataWidth = 32;
static constexpr int kDataInt = 12;
static constexpr int kPackets = kBusWidth / kDataWidth; //16
static constexpr int kShiftData = 2; // Packets 4
static constexpr int MAX_DIM_SIZE = 11008;

typedef union {
  float f;
  uint32_t i;
} d32;

using RawDataT = ap_uint<kBusWidth>;
using StreamT = hls::stream<RawDataT>;
using DataT = ap_fixed<kDataWidth, kDataInt>;

extern "C" {
void matmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols);
}

#endif // __MATMUL_H__
