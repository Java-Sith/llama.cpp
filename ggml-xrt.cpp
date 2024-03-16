#include "ggml-xrt.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// Function to allocate memory for a matrix of size rows x cols
float** allocateMatrix(int rows, int cols) {
    float** matrix = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new float[cols];
    }
    return matrix;
}

// Function to deallocate memory for a matrix
void deallocateMatrix(float** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Function to initialize a matrix with random values
void initializeMatrix(float** matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX; // random float between 0 and 1
        }
    }
}

// Function to perform matrix multiplication
float** matrixMultiplication(float** mat1, float** mat2, int rows1, int cols1, int cols2) {
    float** result = allocateMatrix(rows1, cols2);

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            result[i][j] = 0.0f;
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}

// Function to print a matrix
void printMatrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void quantize_row_q4_0_reference(const float * x, block_q4_0 * y, int k) {
  static const int qk = QK4_0;
  assert(k % qk == 0);
  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
      float amax = 0.0f; // absolute max
      float max  = 0.0f;

      for (int j = 0; j < qk; j++) {
          const float v = x[i*qk + j];
          if (amax < fabsf(v)) {
              amax = fabsf(v);
              max  = v;
          }
      }

      const float d  = max / -8;
      const float id = d ? 1.0f/d : 0.0f;

      y[i].d = ggml_fp32_to_fp16(d);

      for (int j = 0; j < qk/2; ++j) {
          const float x0 = x[i*qk + 0    + j]*id;
          const float x1 = x[i*qk + qk/2 + j]*id;

          const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
          const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

          y[i].qs[j]  = xi0;
          y[i].qs[j] |= xi1 << 4;
    }
  }
}

void quantize_row_q4_0(const float * x, block_q4_0 * y, int k) {
    quantize_row_q4_0_reference(x, y, k);
}

void dequantize_row_q4_0(const block_q4_0 * x, float * y, int k) {
    static const int qk = QK4_0;
    assert(k % qk == 0);
    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}
