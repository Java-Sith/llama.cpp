/*
 * Copyright 2022-2024
 * Author: Luis D. Prieto-Sibaja <prieto.luisdaniel@gmail.com>
 */

#include "dequantize4.h"

// Load function that streams block_q4_K structures into the input stream
void load_input(const block_q4_K *x, hls::stream<block_q4_K> &input_stream, uint64_t k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        #pragma HLS PIPELINE
        input_stream.write(x[i]);
    }
}

#pragma HLS inline
void get_scale_min_k4(int j, const ap_uint<8> *q, ap_uint<6> *d, ap_uint<6> *m) {
    if (j < 4) {
        *d = q[j].range(5, 0);       // Extract lower 6 bits from q[j]
        *m = q[j + 4].range(5, 0);   // Extract lower 6 bits from q[j + 4]
    } else {
        *d = q[j + 4].range(3, 0) | (q[j - 4].range(7, 6) << 4);  // Extract 4 bits + 2 shifted bits
        *m = q[j + 4].range(7, 4) | (q[j].range(7, 6) << 4);      // Extract 4 bits + 2 shifted bits
    }
}

// Convert half-precision to full-precision in HLS
float half_to_float(ap_fixed<16, 5> half_val) {
    return half_val.to_float();  // Converts ap_fixed 16-bit to float
}

// Dequantize function that processes the streamed data
void dequantize(hls::stream<block_q4_K> &input_stream, hls::stream<float> &output_stream, uint64_t k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        #pragma HLS PIPELINE
        block_q4_K block = input_stream.read();

        const ap_fixed<16, 5> d = block.d;
        const ap_fixed<16, 5> dmin = block.dmin;
        const ap_uint<8> *q = block.qs;

        int is = 0;
        ap_uint<6> sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            #pragma HLS UNROLL
            get_scale_min_k4(is, block.scales, &sc, &m);
            const float d1 = half_to_float(d) * sc;
            const float m1 = half_to_float(dmin) * m;
            get_scale_min_k4(is + 1, block.scales, &sc, &m);
            const float d2 = half_to_float(d) * sc;
            const float m2 = half_to_float(dmin) * m;

            for (int l = 0; l < 32; ++l) {
                #pragma HLS UNROLL
                output_stream.write(d1 * (q[l] & 0xF) - m1);  // Lower 4 bits
                output_stream.write(d2 * (q[l] >> 4) - m2);  // Upper 4 bits
            }
            q += 32;
            is += 2;
        }
    }
}

// Store function that streams the output results to memory
void store_result(float *y, hls::stream<float> &output_stream, uint64_t k) {
    const int nb = k / QK_K;
    for (int i = 0; i < nb * QK_K; ++i) {
        #pragma HLS PIPELINE
        y[i] = output_stream.read();
    }
}

// Dequantization kernel
extern "C" {
void dequantize4(block_q4_K *x, float *y, uint64_t k) {
    #pragma HLS INTERFACE m_axi port=x bundle=gmem0
    #pragma HLS INTERFACE m_axi port=y bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=k
    #pragma HLS INTERFACE s_axilite port=return

    hls::stream<block_q4_K> input_stream;
    hls::stream<float> output_stream;

    #pragma HLS STREAM variable=input_stream depth=32
    #pragma HLS STREAM variable=output_stream depth=32
    #pragma HLS dataflow

    // Stream in the block_q4_K structures from memory
    load_input(x, input_stream, k);

    // Dequantize the stream
    dequantize(input_stream, output_stream, k);

    // Stream the results back to memory
    store_result(y, output_stream, k);

}
}
