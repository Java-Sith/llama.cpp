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
                 const int c_tile_row_start = tile_row * tile_size;
                 const int c_tile_col_start = tile_col * tile_size;

                 // Initialize accumulator for the current tile
                 DataT accumulator[tile_size][tile_size];
 #pragma HLS array_partition variable=accumulator complete dim=0

                 // Perform matrix multiplication for the current tile
                 for (int i = 0; i < tile_size; ++i) {
                     for (int j = 0; j < tile_size; ++j) {
 #pragma HLS PIPELINE II=1
                         // Initialize accumulator element
                         accumulator[i][j] = 0;

                         // Perform dot product for elements within the tile
                         for (int k = 0; k < tile_size; ++k) {
 #pragma HLS UNROLL
                             RawDataT a_raw = a.read();
                             RawDataT b_raw = b.read();

                             // Extract data from raw input
                             DataT a_val, b_val;
                             a_val.V = a_raw((k + 1) * 16 - 1, k * 16);
                             b_val.V = b_raw((k + 1) * 16 - 1, k * 16);

                             // Perform multiplication and accumulation
                             accumulator[i][j] += a_val * b_val;
                         }
                     }
                 }

                 // Write the accumulated result to the output stream
                 for (int i = 0; i < tile_size; ++i) {
                     for (int j = 0; j < tile_size; ++j) {
 #pragma HLS PIPELINE II=1
                         // Write the result to the output stream
                         RawDataT c_raw;
                         c_raw.V = accumulator[i][j].V;
                         c.write(c_raw);
                     }
                 }
             }
         }
     }
 }


 static void load_data(RawDataT *a, RawDataT *b, StreamT &a_s, StreamT &b_s,
                       int a_rows, int b_cols, int c_cols) {
 #pragma HLS inline off
     // Load B
     for (int ay = 0; ay < a_rows; ++ay) {
 #pragma HLS pipeline
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
 #pragma HLS pipeline
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
 #pragma HLS inline off
     // Store C
     for (int cy = 0; cy < a_rows; ++cy) {
 #pragma HLS pipeline
         for (int cx = 0; cx < (c_cols >> kShiftData); ++cx) {
             int cidx = cx + cy * (c_cols >> kShiftData);
             // Read data from stream 'c_s' and store in array 'c'
             c[cidx] = c_s.read();
         }
     }
 }

#else
static void matmul_accel(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols) {
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
                const int c_tile_row_start = tile_row * tile_size;
                const int c_tile_col_start = tile_col * tile_size;

                // Perform matrix multiplication for the current tile
                for (int i = 0; i < tile_size; ++i) {
                    for (int j = 0; j < tile_size; ++j) {
#pragma HLS PIPELINE II=1
                        // Initialize accumulator element
                        DataT accumulator = 0;

                        // Perform dot product for elements within the tile
                        for (int k = 0; k < tile_size; ++k) {
#pragma HLS UNROLL
                            // Access elements from input matrices 'a' and 'b'
                            RawDataT a_raw = a[(a_tile_row_start + i) * (b_cols >> kShiftData) + (a_tile_col_start + k)];
                            RawDataT b_raw = b[(b_tile_row_start + k) * (b_cols >> kShiftData) + (b_tile_col_start + j)];

                            // Extract data from raw input
                            DataT a_val, b_val;
                            a_val.V = a_raw((k + 1) * 16 - 1, k * 16);
                            b_val.V = b_raw((k + 1) * 16 - 1, k * 16);

                            // Perform multiplication and accumulation
                            accumulator += a_val * b_val;
                        }

                        // Write the accumulated result to the output matrix 'c'
                        c[(c_tile_row_start + i) * (c_cols >> kShiftData) + (c_tile_col_start + j)] = accumulator;
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
