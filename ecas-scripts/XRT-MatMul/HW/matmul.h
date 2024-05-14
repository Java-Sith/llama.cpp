#ifndef __MATMUL_H__
#define __MATMUL_H__

#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

static constexpr int kBusWidth = 64;
static constexpr int kDataWidth = 16;
static constexpr int kDataInt = 6;
static constexpr int kPackets = kBusWidth / kDataWidth;
static constexpr int kShiftData = 2; // Packets 4

using RawDataT = ap_uint<kBusWidth>;
using StreamT = hls::stream<RawDataT>;
using DataT = ap_fixed<kDataWidth, kDataInt>;

extern "C" {
void matmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols);
}

#endif // __MATMUL_H__
