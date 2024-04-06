#include "ggml-xrt.h"
#include <vector>
#include <iostream>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

const int M = 1280;
const int N = 1536;
const int K = 1280;

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

/*void quantize_row_q4_0_reference(const float * x, block_q4_0 * y, int k) {
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
}*/

void ggml_xrt_mul_mat(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
  float** matrix1 = allocateMatrix(M, K);
  float** matrix2 = allocateMatrix(K, N);
  initializeMatrix(matrix1, M, K);
  initializeMatrix(matrix2, K, N);
  float** resultMatrix = matrixMultiplication(matrix1, matrix2, M, K, N);
  std::cout << "Resultant Matrix:" << std::endl;
  printMatrix(resultMatrix, M, N);
  deallocateMatrix(matrix1, M);
  deallocateMatrix(matrix2, K);
  deallocateMatrix(resultMatrix, M);
}

void ggml_init_xrt() {
    static bool initialized = false;
    if (!initialized)
    {
        initialized = true;
    }
    printf("%d", initialized); // prints 1
    printf("\n");
}

bool ggml_xrt_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    ggml_xrt_func_t func;
    if (tensor->op == GGML_OP_MUL_MAT) {
       if (tensor->src[0]->ne[3] != tensor->src[1]->ne[3]) {
#ifndef NDEBUG
           fprintf(stderr, "%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, tensor->name, tensor->src[0]->ne[3], tensor->src[1]->ne[3]);
#endif
           return false;
       }
   }
   switch (tensor->op) {
        case GGML_OP_REPEAT:
            //func = ggml_xrt_repeat;
            //break;
        case GGML_OP_GET_ROWS:
            //func = ggml_xrt_get_rows;
            //break;
        case GGML_OP_DUP:
            //func = ggml_xrt_dup;
            //break;
        case GGML_OP_ADD:
            //func = ggml_xrt_add;
            //break;
        case GGML_OP_ACC:
            //func = ggml_xrt_acc;
            //break;
        case GGML_OP_MUL:
            //func = ggml_xrt_mul;
            //break;
        case GGML_OP_DIV:
            //func = ggml_xrt_div;
            //break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(tensor)) {
                case GGML_UNARY_OP_GELU:
                    //func = ggml_xrt_gelu;
                    //break;
                case GGML_UNARY_OP_SILU:
                    //func = ggml_xrt_silu;
                    //break;
                case GGML_UNARY_OP_GELU_QUICK:
                    //func = ggml_xrt_gelu_quick;
                    //break;
                case GGML_UNARY_OP_TANH:
                    //func = ggml_xrt_tanh;
                    //break;
                case GGML_UNARY_OP_RELU:
                    //func = ggml_xrt_relu;
                    //break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            //func = ggml_xrt_norm;
            //break;
        case GGML_OP_GROUP_NORM:
            //func = ggml_xrt_group_norm;
            //break;
        case GGML_OP_CONCAT:
            //func = ggml_xrt_concat;
            //break;
        case GGML_OP_UPSCALE:
            //func = ggml_xrt_upscale;
            //break;
        case GGML_OP_PAD:
            //func = ggml_xrt_pad;
            //break;
        case GGML_OP_LEAKY_RELU:
            //func = ggml_xrt_leaky_relu;
            //break;
        case GGML_OP_RMS_NORM:
            //func = ggml_xrt_rms_norm;
            //break;
        case GGML_OP_MUL_MAT:
            func = ggml_xrt_mul_mat;
            break;
        case GGML_OP_MUL_MAT_ID:
            func = ggml_xrt_mul_mat;
            break;
        case GGML_OP_SCALE:
            //func = ggml_xrt_scale;
            //break;
        case GGML_OP_SQR:
            //func = ggml_xrt_sqr;
            //break;
        case GGML_OP_CLAMP:
            //func = ggml_xrt_clamp;
            //break;
        case GGML_OP_CPY:
            //func = ggml_xrt_cpy;
            //break;
        case GGML_OP_CONT:
            //func = ggml_xrt_dup;
            //break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            //func = ggml_xrt_nop;
            //break;
        case GGML_OP_DIAG_MASK_INF:
            //func = ggml_xrt_diag_mask_inf;
            //break;
        case GGML_OP_SOFT_MAX:
            //func = ggml_xrt_soft_max;
            //break;
        case GGML_OP_ROPE:
            //func = ggml_xrt_rope;
            //break;
        case GGML_OP_ALIBI:
            //func = ggml_xrt_alibi;
            //break;
        case GGML_OP_IM2COL:
            //func = ggml_xrt_im2col;
            //break;
        case GGML_OP_SUM_ROWS:
            //func = ggml_xrt_sum_rows;
            //break;
        case GGML_OP_ARGSORT:
            //func = ggml_xrt_argsort;
            //break;
        default:
            return false;
    }
    if (params->ith != 0) {
        return true;
    }
    func(tensor->src[0], tensor->src[1], tensor);
    return true;
}
