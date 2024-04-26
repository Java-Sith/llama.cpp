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
   DataT a_tile[TILE_SIZE][TILE_SIZE];
   DataT b_tile[TILE_SIZE][TILE_SIZE];

 matmul_samples:
   for (int ay = 0; ay < a_rows; ay += TILE_SIZE) {
     for (int bx = 0; bx < b_cols; bx += TILE_SIZE) {
       for (int cx = 0; cx < c_cols; cx += TILE_SIZE) {
 #pragma HLS pipeline
         // Load tiles from matrices a and b
         for (int i = 0; i < TILE_SIZE; ++i) {
           for (int j = 0; j < TILE_SIZE; ++j) {
             a_tile[i][j] = a.read();
             b_tile[i][j] = b.read();
           }
         }

         // Perform multiplication on the tiles
         for (int i = 0; i < TILE_SIZE; ++i) {
           for (int j = 0; j < TILE_SIZE; ++j) {
             DataT val = 0.f;
             for (int k = 0; k < TILE_SIZE; ++k) {
               val += a_tile[i][k] * b_tile[k][j];
             }

             // Get the indices
             int cx_p_1 = cx + j + 1;
             int val_mod = cx_p_1 & (kPackets - 1);
             int cx_mod = (cx + j) & (kPackets - 1);

             // Write accordingly
             int poff_low = cx_mod * kDataWidth;
             int poff_high = poff_low + kDataWidth - 1;

             RawDataT valpacket;
             valpacket(poff_high, poff_low) = val.V;

             // Stream out if done
             if (val_mod == 0) {
               c.write(valpacket);
             }
           }
         }
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
  for (int ay = 0; ay < a_rows; ++ay) {
    for (int cx = 0; cx < c_cols; ++cx) {
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

  DataT a_tile[TILE_SIZE][TILE_SIZE];
  DataT b_tile[TILE_SIZE][TILE_SIZE];

matmul_samples:
  for (int ay = 0; ay < a_rows; ay += TILE_SIZE) {
    for (int bx = 0; bx < b_cols_shift; bx += TILE_SIZE) {
      for (int cx = 0; cx < c_cols; cx += TILE_SIZE) {
#pragma HLS pipeline
        // Load tiles from matrices a and b
        for (int i = 0; i < TILE_SIZE; ++i) {
          for (int j = 0; j < TILE_SIZE; ++j) {
            a_tile[i][j] = a[(ay + i) * b_cols_shift + bx + j];
            b_tile[i][j] = b[(bx + i) * c_cols_shift + cx + j];
          }
        }

        // Perform multiplication on the tiles
        for (int i = 0; i < TILE_SIZE; ++i) {
          for (int j = 0; j < TILE_SIZE; ++j) {
            DataT val = 0.f;
            for (int k = 0; k < TILE_SIZE; ++k) {
              val += a_tile[i][k] * b_tile[k][j];
            }
            c[(ay + i) * c_cols_shift + cx + j] += val;
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
  matmul_accel(a, b, c, a_rows, b_cols, c_cols);
#endif

}

}
