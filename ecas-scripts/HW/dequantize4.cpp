/*
 * Copyright 2022-2024
 * Author: Luis D. Prieto-Sibaja <prieto.luisdaniel@gmail.com>
 */

#include "dequantize4.h"

struct block_q4_K {
    half d;  // HLS half for 16-bit float
    uint8_t ql[32];  // Quantized low bits
    uint8_t qh[16];  // Quantized high bits
};

static float half_to_float(half val) {
    return (float)val;  // Implicit conversion in HLS
}

static int kSubBlocks;
static int kTotalMaxSize;

void initialize_constants(int subblocks, int maxsize) {
    kSubBlocks = subblocks;
    kTotalMaxSize = maxsize;
}

static void load_input(block_q4_K *in, hls::stream<block_q4_K> &inStream, uint64_t size) {
    const uint64_t size_raw = size / sizeof(block_q4_K);
mem_rd:
    for (uint64_t i = 0; i < size_raw; ++i) {
#pragma HLS PIPELINE
        inStream << in[i];
    }
}

static void store_output(float *out, hls::stream<float> &outStream, uint64_t size) {
    const uint64_t size_raw = size / sizeof(float);
mem_wr:
    for (uint64_t i = 0; i < size_raw; ++i) {
#pragma HLS PIPELINE
        out[i] = outStream.read();
    }
}

// Perform dequantization for a block of Q4 data and write to output stream
static void dequantize_block(hls::stream<block_q4_K> &input_stream,
                              hls::stream<float> &output_stream,
                              uint64_t size) {
    const uint64_t num_blocks = size / kPackets;

block_loop:
    for (uint64_t i = 0; i < num_blocks; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg = kTotalMaxSize
        block_q4_K block = input_stream.read();

    subblock_loop:
        for (int l = 0; l < kSubBlocks; ++l) {
#pragma HLS UNROLL
            float dequantized_value;
            
            int8_t ql_val = (int8_t)(block.ql[l] & 0xF) - 32;
            int8_t qh_val = (int8_t)((block.qh[l] >> 0) & 3);

            float d = (float)block.d;
            dequantized_value = ((ql_val | (qh_val << 4)) * d);

            output_stream << dequantized_value;
        }
    }
}


extern "C" {
void dequantize4(block_q4_K *in, float *out, uint64_t size) {
#pragma HLS INTERFACE m_axi port=in bundle=gmem0
#pragma HLS INTERFACE m_axi port=out bundle=gmem1
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=return

    hls::stream<block_q4_K> in_stream("in_stream");
    hls::stream<float> out_stream("out_stream");

#pragma HLS stream variable=in_stream depth=32
#pragma HLS stream variable=out_stream depth=32
#pragma HLS dataflow

    // Initialize the constants
    initialize_constants(size / QK_K, size);

    // Load input blocks
    load_input(in, in_stream, size);

    // Dequantize each block and convert to float
    dequantize_block(in_stream, out_stream, size);

    // Store the result in output memory
    store_output(out, out_stream, size);
}
}
