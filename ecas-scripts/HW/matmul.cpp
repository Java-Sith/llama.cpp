#include "config.h"
#include <algorithm>

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

static void load_data(RawDataT *a, RawDataT *b, RawDataT arrA[], RawDataT arrB[],
               int a_rows, int b_cols, int c_cols) {
  // Load B
  for (int ay = 0; ay < a_rows; ++ay) {
#pragma HLS pipeline
    for (int cx = 0; cx < c_cols; ++cx) {
      for (int bx = 0; bx < (b_cols >> kShiftData); ++bx) {
        int bidx = bx + cx * (b_cols >> kShiftData);
        arrB[bidx] = b[bidx];
      }
    }
  }

  // Load A
  for (int cx = 0; cx < c_cols; ++cx) {
    for (int ay = 0; ay < a_rows; ++ay) {
#pragma HLS pipeline
      for (int ax = 0; ax < (b_cols >> kShiftData); ++ax) {
        int aidx = ax + ay * (b_cols >> kShiftData);
        arrA[aidx] = a[aidx];
      }
    }
  }
}

static void store_data(RawDataT *c, RawDataT arrC[],
               int a_rows, int b_cols, int c_cols) {

  // Load C
  for (int cy = 0; cy < a_rows; ++cy) {
#pragma HLS pipeline
    for (int cx = 0; cx < (c_cols >> kShiftData); ++cx) {
      int cidx = cx + cy * (c_cols >> kShiftData);
      c[cidx] = arrC[cidx];
    }
  }
}

static void matmul_accel (RawDataT *arrA, RawDataT *arrB, RawDataT *arrC, int a_rows, int b_cols, int c_cols) {
  int b_cols_shift = b_cols >> kShiftData;
  int c_cols_shift = c_cols >> kShiftData;

  for (int ay_tile = 0; ay_tile < a_rows; ay_tile += TILE_SIZE) {
    for (int cx_tile = 0; cx_tile < c_cols; cx_tile += TILE_SIZE) {
      for (int bx_tile = 0; bx_tile < b_cols_shift; bx_tile += TILE_SIZE) {
#pragma HLS pipeline
        for (int ay = ay_tile; ay < std::min(ay_tile + TILE_SIZE, a_rows); ++ay) {
          RawDataT valpacket = 0;
          for (int cx = cx_tile; cx < std::min(cx_tile + TILE_SIZE, c_cols); ++cx) {
#pragma HLS pipeline
            DataT val = 0.f;
            for (int bx = bx_tile; bx < std::min(bx_tile + TILE_SIZE, b_cols_shift); ++bx) {
#pragma HLS unroll
              RawDataT a_raw = arrA[ay * b_cols_shift + bx];
              RawDataT b_raw = arrB[cx * b_cols_shift + bx];
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
            // Write accordingly 
            int poff_low = cx_mod * kDataWidth;
            int poff_high = poff_low + kDataWidth - 1;
            valpacket(poff_high, poff_low) = val.V;
            // Stream out if done
            if (val_mod == 0) {
              arrC[cx_div + ay * c_cols_shift] = valpacket;
              valpacket = 0;
            }
          }
        }
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
  // Define array size and URAM access
  int size_a = a_rows * b_cols;
  int size_b = c_cols * b_cols;
  int size_c = a_rows * c_cols;

  RawDataT localA[size_a];
  RawDataT localB[size_b];
  RawDataT localC[size_c];

#pragma HLS ARRAY_PARTITION block variable=localA factor=partitions
#pragma HLS ARRAY_PARTITION block variable=localB factor=partitions
#pragma HLS ARRAY_PARTITION block variable=localC factor=partitions

#pragma HLS resource variable=localA core=XPM_MEMORY uram
#pragma HLS resource variable=localB core=XPM_MEMORY uram
#pragma HLS resource variable=localC core=XPM_MEMORY uram

readIn:
  for (int i = 0; i < partitions; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = size_a max = size_b
    load_data(&a[size_a*i/partitions], &b[size_b*i/partitions], &localA[size_a*i/partitions], 
            &localB[size_b*i/partitions], a_rows, b_cols, c_cols);
  }


#pragma HLS DATAFLOW
matmul_loop:
  for (int i=0; i < partitions; ++i) {
    #pragma HLS UNROLL
    // each instance accesses a different block
    matmul_accel(&localA[size_a*i/partitions], &localB[size_b*i/partitions], &localC[size_c*i/partitions],
                a_rows, b_cols, c_cols);
  }

writeOut:
  for (int i = 0; i < partitions; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = size_c max = size_c
    store_data(&c[size_c*i/partitions], &localC[size_c*i/partitions], a_rows, b_cols, c_cols);
  }

#endif

}

}
