#include "ggml.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#if defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

#if defined(GGML_GQ_USE_FP16_SCALE)
#define gq_scale_t ggml_fp16_t
#define GGML_FP32_TO_GQ(x) ggml_fp32_to_fp16(x)
#define GGML_GQ_TO_FP32(x) ggml_fp16_to_fp32(x)
#else
#define gq_scale_t float
#define GGML_FP32_TO_GQ(x) (x)
#define GGML_GQ_TO_FP32(x) (x)
#endif

const int M = 1280;
const int N = 1536;
const int K = 1280;

#define QK 64
#define QB 4
#define gq_t_bits 64
#define gq_quant_t uint64_t

//Naive implementation of Mul Mat
void mul_mat(float* restrict src0, float* restrict src1, float *dst, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += src0[i*k + l] * src1[j*k + l];
            }
            dst[i*n + j] = sum;
        }
    }
}

static inline int quantize_4_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_4_row_size(int k) {
    const int nb = quantize_4_blocks_per_row(k);

    return nb*(2*sizeof(gq_scale_t) + QK/2);
}

void vec_dot_gq_4 (const int n, float * restrict s, const float * restrict x, const float * restrict y) {
    const int nb = quantize_4_blocks_per_row(n);

    const gq_scale_t * restrict pm0 = (const gq_scale_t *) x;
    const gq_scale_t * restrict pm1 = (const gq_scale_t *) y;

    const gq_scale_t * restrict pd0 = pm0 + nb;
    const gq_scale_t * restrict pd1 = pm1 + nb;

    const uint8_t * restrict pb0 = (const uint8_t *) (pd0 + nb);
    const uint8_t * restrict pb1 = (const uint8_t *) (pd1 + nb);

    float sumf = 0.0;

    for (int i = 0; i < nb; i++) {
        const float m0 = GGML_GQ_TO_FP32(pm0[i]);
        const float d0 = GGML_GQ_TO_FP32(pd0[i]);

        const float m1 = GGML_GQ_TO_FP32(pm1[i]);
        const float d1 = GGML_GQ_TO_FP32(pd1[i]);

        const uint8_t * restrict p0 = pb0 + i*QK/2;
        const uint8_t * restrict p1 = pb1 + i*QK/2;

        for (int j = 0; j < QK/2; j++) {
            const uint8_t v0 = p0[j];
            const uint8_t v1 = p1[j];

            const float f0 = d0*(v0 & 0xf) + m0;
            const float f1 = d0*(v0 >> 4)  + m0;

            const float f2 = d1*(v1 & 0xf) + m1;
            const float f3 = d1*(v1 >> 4)  + m1;

            sumf += f0*f2 + f1*f3;
        }
    }
    *s = sumf;
}

void mul_mat_gq_4(const void * src0, const float * src1, float * dst, int m, int n, int k) {
    assert(k % QK == 0);
    const int nb = quantize_4_blocks_per_row(k);
    for (int ir0 = 0; ir0 < m; ir0++) {
        for (int ir1 = 0; ir1 < n; ir1++) {
            vec_dot_gq_4(k, dst + ir1, src0, src1);
            src1 = (const char *) src1 + quantize_4_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_4_row_size(k);
        src1 = (const char *) src1 - n*quantize_4_row_size(k);
        dst = (float *) dst + n;
    }
}

gq_scale_t *load_tensor(const char *filename, int rows, int cols) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    // Allocate memory for the tensor structure
    float * tensor = malloc(sizeof(float)*rows*cols);
    if (tensor == NULL) {
        perror("Memory allocation error");
        exit(EXIT_FAILURE);
    }

    // Read tensor data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%f", &(tensor[i * cols + j])) != 1) {
                fprintf(stderr, "Error reading tensor data from file\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);

    return tensor;
}

void save_tensor(const char *filename, gq_scale_t *tensor, int rows, int cols) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    // Write tensor data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%.6f ", tensor[i * cols + j]); // Adjust precision as needed
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main(int argc, const char ** argv) {
    assert(sizeof(gq_quant_t)*8 == gq_t_bits);
    ggml_time_init();

    float * src0 = load_tensor("~/llama.cpp/ecas-scripts/tensor1.txt", M, K);
    float * src1  = load_tensor("~/llama.cpp/ecas-scripts/tensor2.txt", K, N);
    float * dst  = malloc(sizeof(float)*M*N);

    double iM = 1.0/M;
    double sum = 0.0f;

    int method = 0;
    if (argc > 1) {
        method = atoi(argv[1]);
    }

    const int64_t start = ggml_cycles();
    const int64_t start_us = ggml_time_us();

    if (method == 0) {
        #ifdef GGML_USE_OPENBLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, src0, M, src1, N, 0.0f, dst, M);
            save_tensor("~/llama.cpp/ecas-scripts/result.txt", (gq_scale_t *) dst, M, N);
        #else
            mul_mat(src0, src1, dst, M, N, K);
            save_tensor("~/llama.cpp/ecas-scripts/result.txt", (gq_scale_t *) dst, M, N);
        #endif
    }

    if (method == 1) {
        #ifdef GGML_USE_OPENBLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, src0, M, src1, N, 0.0f, dst, M);
            save_tensor("~/llama.cpp/ecas-scripts/result.txt", (gq_scale_t *) dst, M, N);
        #else
            mul_mat_gq_4(src0, src1, dst, M, N, K);
            save_tensor("~/llama.cpp/ecas-scripts/result.txt", (gq_scale_t *) dst, M, N);
        #endif
    }
    for (int i = 0; i < N; i++) {
        sum += dst[i]*iM;
    }

    for (int i = 0; i < 16; ++i) {
        printf("%f\n", dst[i]);
    }

    const int64_t end = ggml_cycles();
    const int64_t end_us = ggml_time_us();
    printf("%s: elapsed ticks: %" PRIu64 "\n",  __func__, end - start);
    printf("%s: elapsed us:    %d / %f ms\n",  __func__, (int)(end_us - start_us), (end_us - start_us) / 1000.0);
    printf("%f\n", sum);

    free(src0);
    free(src1);
    free(dst);

    return 0;
}
