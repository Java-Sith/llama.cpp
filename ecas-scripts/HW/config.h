#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

static constexpr int kBusWidth = 64;
static constexpr int kDataWidth = 16;
static constexpr int kDataInt = 6;
static constexpr int kPackets = kBusWidth / kDataWidth;
static constexpr int kShiftData = 2; // Packets 4
static constexpr int TILE_SIZE = 16;

using RawDataT = ap_uint<kBusWidth>;
using StreamT = hls::stream<RawDataT>;
using DataT = ap_fixed<kDataWidth, kDataInt>;

#endif // __CONFIG_H__
