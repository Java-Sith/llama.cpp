#include "config.h"

static void matmul_gemm(StreamT &a, StreamT &b, StreamT &c, const int a_rows,
                        const int b_cols, const int c_cols) {
#pragma HLS INLINE off

  for (int c_row = 0; c_row < a_rows; ++c_row) { // m
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
    for (int c_col = 0; c_col < c_cols; c_col += kPackets) { // n
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
      RawDataT c_packet = 0;
    
      for (int c_p = 0; c_p < kPackets; ++c_p) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
        float c_val = 0.f;                  
        for (int b_col = 0; b_col < b_cols; b_col += kPackets) { // k
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
          RawDataT a_packet = a.read();
          RawDataT b_packet = b.read();
          float res = 0.f; 
          // Decompose packets
          for (int p = 0; p < kPackets; ++p) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=32 avg=32
#pragma HLS UNROLL
            const int low = p * kDataWidth;
            const int high = low + kDataWidth - 1;
            d32 a_val;
            a_val.i = a_packet(high, low);
            d32 b_val;
            b_val.i = b_packet(high, low);
            res += a_val.f * b_val.f;
          }
          c_val += res;
        }
        const int low = c_p * kDataWidth;
        const int high = low + kDataWidth - 1;
        c_packet(high, low) = c_val;
      }

      c.write(c_packet);
    }
  }
}

static void matmul_to_stream_a(RawDataT *a, StreamT &sa, const int rows,
                             const int cols, const int rep_rows,
                             const int rep_mats) {
#pragma HLS INLINE off
  const int tcols = cols / kPackets;

  for (int rep_mat = 0; rep_mat < rep_mats; ++rep_mat) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
    // Repeated matrix transmission
    for (int row = 0; row < rows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
      // Repeated row transmission
      for (int rep_row = 0; rep_row < rep_rows; ++rep_row) {
#pragma HLS LOOP_TRIPCOUNT min=4096 max=4096 avg=4096
        // Transmit columns
        for (int col = 0; col < tcols; ++col) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE
          const int row_shift = row * tcols;
          const int cols_shift = col;
          const int shift = cols_shift + row_shift;          
          RawDataT packet = a[shift];
          sa.write(packet);
        }
      }
    }
  }
}

static void matmul_to_stream_b(RawDataT *a, StreamT &sa, const int rows,
                             const int cols, const int rep_rows,
                             const int rep_mats) {
#pragma HLS INLINE off
  const int tcols = cols / kPackets;

  for (int rep_mat = 0; rep_mat < rep_mats; ++rep_mat) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2 avg=2
    // Repeated matrix transmission
    for (int row = 0; row < rows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=4096 max=4096 avg=4096
      // Repeated row transmission
      for (int rep_row = 0; rep_row < rep_rows; ++rep_row) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1 avg=1
        // Transmit columns
        for (int col = 0; col < tcols; ++col) {
#pragma HLS LOOP_TRIPCOUNT min=128 max=128 avg=128
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE
          const int row_shift = row * tcols;
          const int cols_shift = col;
          const int shift = cols_shift + row_shift;          
          RawDataT packet = a[shift];
          sa.write(packet);
        }
      }
    }
  }
}

static void matmul_from_stream(RawDataT *a, StreamT &sa, const int length) {
#pragma HLS INLINE off
  const int tlength = length / kPackets;
  for (int i = 0; i < tlength; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_TRIPCOUNT min=256 max=256 avg=256
    a[i] = sa.read();
  }
}

extern "C" {

/**
 * matrix: (rows, cols)
 * a: input (samples, inputs)
 * b: weights (outputs, inputs) assumed transposed
 * c: output (samples, outputs)
 */
void matmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols,
            int c_cols) {
#pragma HLS INTERFACE m_axi offset = slave port = a bundle = gmem0 depth = 128
#pragma HLS INTERFACE m_axi offset = slave port = b bundle = gmem1 depth = 128
#pragma HLS INTERFACE m_axi offset = slave port = c bundle = gmem2 depth = 128
#pragma HLS INTERFACE s_axilite register port = a_rows
#pragma HLS INTERFACE s_axilite register port = b_cols
#pragma HLS INTERFACE s_axilite register port = c_cols
#pragma HLS INTERFACE s_axilite register port = return

  static StreamT stream_a;
  static StreamT stream_b;
  static StreamT stream_c;
#pragma HLS stream variable = stream_a depth = MAX_DIM_SIZE
#pragma HLS stream variable = stream_b depth = MAX_DIM_SIZE
#pragma HLS stream variable = stream_c depth = MAX_DIM_SIZE

#pragma HLS dataflow
  matmul_to_stream_a(a, stream_a, a_rows, b_cols, c_cols, 1);
  matmul_to_stream_b(b, stream_b, b_cols, c_cols, 1, a_rows);
  matmul_gemm(stream_a, stream_b, stream_c, a_rows, b_cols, c_cols);
  matmul_from_stream(c, stream_c, c_cols * a_rows);
}
}
