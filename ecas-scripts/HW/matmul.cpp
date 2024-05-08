#include "config.h"

#ifdef USE_AXI_STREAM
/**
 * a: input (samples, inputs)
 * b: weights (outputs, inputs)
 * c: output (samples, outputs)
 *
 * here:
 * a_rows = c_rows -> samples
 * a_cols = b_cols -> inputs
 * b_rows = c_cols -> outputs
 */
static void matmul_accel (StreamT &a, StreamT &b, StreamT &c, int a_rows, int b_cols, int c_cols) {

matmul_samples:
  for (int ay = 0; ay < a_rows; ++ay) {
matmul_layers:
    RawDataT valpacket = 0;
    for (int cx = 0; cx < c_cols; ++cx) {
#pragma HLS pipeline
      DataT val = 0.f;
matmul_perceptron:
      for (int bx = 0; bx < b_cols; bx += kPackets) {
        RawDataT a_raw = a.read();
        RawDataT b_raw = b.read();
        for (int p = 0; p < kPackets; ++p) {
#pragma HLS unroll
          int poff_low = p * kDataWidth;
          int poff_high = poff_low + kDataWidth - 1;
          
          DataT a, b;

          a.V = a_raw(poff_high, poff_low);
          b.V = b_raw(poff_high, poff_low);

          val += a * b;
        }
      }
      // Get the indices
      int cx_p_1 = cx + 1;
      int val_mod = cx_p_1 & (kPackets - 1);
      int cx_mod = cx & (kPackets - 1);

      // Write accordingly 
      int poff_low = cx_mod * kDataWidth;
      int poff_high = poff_low + kDataWidth - 1;

      valpacket(poff_high, poff_low) = val.V;
      
      // Stream out if done
      if (val_mod == 0) {
        c.write(valpacket);
        valpacket = 0;
      }
    }
  }
}

static void load_data(RawDataT *a, RawDataT *b, StreamT &a_s, StreamT &b_s,
               int a_rows, int b_cols, int c_cols) {
  // Load B
  for (int ay = 0; ay < a_rows; ++ay) {
#pragma HLS pipeline
    for (int cx = 0; cx < c_cols; ++cx) {
      for (int bx = 0; bx < (b_cols >> kShiftData); ++bx) {
        int bidx = bx + cx * (b_cols >> kShiftData);
        b_s.write(b[bidx]);
      }
    }
  }

  // Load A
  for (int cx = 0; cx < c_cols; ++cx) {
    for (int ay = 0; ay < a_rows; ++ay) {
#pragma HLS pipeline
      for (int ax = 0; ax < (b_cols >> kShiftData); ++ax) {
        int aidx = ax + ay * (b_cols >> kShiftData);
        a_s.write(a[aidx]);
      }
    }
  }
}

static void store_data(RawDataT *c, StreamT &c_s,
               int a_rows, int b_cols, int c_cols) {

  // Load C
  for (int cy = 0; cy < a_rows; ++cy) {
#pragma HLS pipeline
    for (int cx = 0; cx < (c_cols >> kShiftData); ++cx) {
      int cidx = cx + cy * (c_cols >> kShiftData);
      c[cidx] = c_s.read();
    }
  }
}
#else
static void matmul_accel (RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols) {
  int b_cols_shift = b_cols >> kShiftData;
  int c_cols_shift = c_cols >> kShiftData;

matmul_samples:
  for (int ay = 0; ay < a_rows; ++ay) {
matmul_layers:
    RawDataT valpacket = 0;
    for (int cx = 0; cx < c_cols; ++cx) {
#pragma HLS pipeline
      DataT val = 0.f;
matmul_perceptron:
      for (int bx = 0; bx < b_cols_shift; ++bx) {
#pragma HLS pipeline
        RawDataT a_raw = a[ay * b_cols_shift + bx];
        RawDataT b_raw = b[cx * b_cols_shift + bx];
        for (int p = 0; p < kPackets; ++p) {
#pragma HLS unroll
          int poff_low = p * kDataWidth;
          int poff_high = poff_low + kDataWidth - 1;
          
          DataT a, b;

          a.V = a_raw(poff_high, poff_low);
          b.V = b_raw(poff_high, poff_low);

          val += a * b;
        }
      }

      // Get the indices
      int cx_mod = cx & (kPackets - 1);
      int cx_div = cx >> kShiftData;
      int val_mod = (cx + 1) & (kPackets - 1);
#pragma HLS pipeline
      // Write accordingly 
      int poff_low = cx_mod * kDataWidth;
      int poff_high = poff_low + kDataWidth - 1;

      valpacket(poff_high, poff_low) = val.V;

      // Stream out if done
      if (val_mod == 0) {
        c[cx_div + ay * c_cols_shift] = valpacket;
        valpacket = 0;
      }
    }
  }
}
#endif

extern "C" {

/**
 * matrix: (rows, cols)
 * a: input (samples, inputs)
 * b: weights (outputs, inputs)
 * c: output (samples, outputs)
 */
void matmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols) {
#pragma HLS INTERFACE m_axi offset=slave port=a bundle=gmem0
#pragma HLS INTERFACE m_axi offset=slave port=b bundle=gmem1
#pragma HLS INTERFACE m_axi offset=slave port=c bundle=gmem2
#pragma HLS INTERFACE s_axilite register port=a_rows
#pragma HLS INTERFACE s_axilite register port=b_cols
#pragma HLS INTERFACE s_axilite register port=c_cols
#pragma HLS INTERFACE s_axilite register port=return

#ifdef USE_AXI_STREAM
  static StreamT a_stream, b_stream, c_stream;
#pragma HLS stream variable = a_stream depth = 128
#pragma HLS stream variable = b_stream depth = 128
#pragma HLS stream variable = c_stream depth = 128

#pragma HLS dataflow
  load_data(a, b, a_stream, b_stream, a_rows, b_cols, c_cols);
  matmul_accel(a_stream, b_stream, c_stream, a_rows, b_cols, c_cols);
  store_data(c, c_stream, a_rows, b_cols, c_cols);

#else
  // Define partition size and number of partitions
  const int partition_size = 64; // Adjust partition size as needed
  const int num_partitions = (a_rows + partition_size - 1) / partition_size;

matmul_loop:
#pragma HLS PIPELINE
  // Loop over partitions of matrices 'a' and 'b'
  for (int partition = 0; partition < num_partitions; ++partition) {
#pragma HLS loop_tripcount min = 1 max = num_partitions
      // Compute indices for current partition
      int start_row = partition * partition_size;
      int end_row = std::min(start_row + partition_size, a_rows);

      // Call matmul_accel for the current partition
      matmul_accel(&a[start_row * b_cols], b, &c[start_row * c_cols], end_row - start_row, b_cols, c_cols);
  }
#endif

}

}
