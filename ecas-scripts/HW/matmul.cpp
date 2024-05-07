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
static void matmul_accel(StreamT &a, StreamT &b, StreamT &c, int a_rows, int b_cols, int c_cols) {
    const int tile_size = 16;
    const int num_tiles = a_rows / tile_size;

    // Loop over tiles
    for (int tile_row = 0; tile_row < num_tiles; ++tile_row) {
        for (int tile_col = 0; tile_col < num_tiles; ++tile_col) {
            // Loop over tiles within a tile row/column
            for (int tile_iter = 0; tile_iter < num_tiles; ++tile_iter) {
                // Compute indices for current tile
                const int a_tile_row_start = tile_row * tile_size;
                const int a_tile_col_start = tile_iter * tile_size;
                const int b_tile_row_start = tile_iter * tile_size;
                const int b_tile_col_start = tile_col * tile_size;

                // Initialize accumulator for the current tile
                DataT accumulator = 0;

                // Perform matrix multiplication for the current tile
                for (int i = 0; i < tile_size; ++i) {
                    for (int j = 0; j < tile_size; ++j) {
#pragma HLS PIPELINE
                        // Perform dot product for elements within the tile
                        for (int k = 0; k < tile_size; ++k) {
#pragma HLS UNROLL
                            // Read elements from input streams 'a' and 'b'
                            RawDataT a_val = a.read();
                            RawDataT b_val = b.read();

                            // Extract data from raw input
                            DataT a_data, b_data;
                            a_data.V = a_val;
                            b_data.V = b_val;

                            // Perform multiplication and accumulation
                            accumulator += a_data * b_data;
                        }
                    }
                }

                // Write the accumulated result to the output stream 'c'
                c.write(accumulator);
            }
        }
    }
}


 static void load_data(RawDataT *a, RawDataT *b, StreamT &a_s, StreamT &b_s,
                       int a_rows, int b_cols, int c_cols) {
 #pragma HLS INLINE OFF
     // Load B
     for (int ay = 0; ay < a_rows; ++ay) {
 #pragma HLS PIPELINE
         for (int cx = 0; cx < c_cols; ++cx) {
             for (int bx = 0; bx < (b_cols >> kShiftData); ++bx) {
                 int bidx = bx + cx * (b_cols >> kShiftData);
                 // Write data to stream 'b_s'
                 b_s.write(b[bidx]);
             }
         }
     }

     // Load A
     for (int cx = 0; cx < c_cols; ++cx) {
         for (int ay = 0; ay < a_rows; ++ay) {
 #pragma HLS PIPELINE
             for (int ax = 0; ax < (b_cols >> kShiftData); ++ax) {
                 int aidx = ax + ay * (b_cols >> kShiftData);
                 // Write data to stream 'a_s'
                 a_s.write(a[aidx]);
             }
         }
     }
 }

 static void store_data(RawDataT *c, StreamT &c_s,
                        int a_rows, int b_cols, int c_cols) {
 #pragma HLS INLINE OFF
     // Store C
     for (int cy = 0; cy < a_rows; ++cy) {
 #pragma HLS PIPELINE
         for (int cx = 0; cx < (c_cols >> kShiftData); ++cx) {
             int cidx = cx + cy * (c_cols >> kShiftData);
             // Read data from stream 'c_s' and store in array 'c'
             c[cidx] = c_s.read();
         }
     }
 }

#else
static void matmul_accel_tiled_optimized(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols) {
  int b_cols_shift = b_cols >> kShiftData;
  int c_cols_shift = c_cols >> kShiftData;

  for (int ay_tile = 0; ay_tile < a_rows; ay_tile += TILE_SIZE) {
    for (int cx_tile = 0; cx_tile < c_cols; cx_tile += TILE_SIZE) {
      for (int bx_tile = 0; bx_tile < b_cols_shift; bx_tile += TILE_SIZE) {
#pragma HLS PIPELINE
        for (int ay = ay_tile; ay < std::min(ay_tile + TILE_SIZE, a_rows); ++ay) {
          RawDataT valpacket = 0;
          for (int cx = cx_tile; cx < std::min(cx_tile + TILE_SIZE, c_cols); ++cx) {
#pragma HLS PIPELINE
            DataT val = 0.f;
            for (int bx = bx_tile; bx < std::min(bx_tile + TILE_SIZE, b_cols_shift); ++bx) {
#pragma HLS UNROLL
              RawDataT a_raw = a[ay * b_cols_shift + bx];
              RawDataT b_raw = b[cx * b_cols_shift + bx];
              for (int p = 0; p < kPackets; ++p) {
#pragma HLS UNROLL
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
              c[cx_div + ay * c_cols_shift] = valpacket;
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
  matmul_accel(a, b, c, a_rows, b_cols, c_cols);
#endif

}

}
