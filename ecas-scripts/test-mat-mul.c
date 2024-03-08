#include "../ggml.h"
#include "../ggml-quants.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef MIN
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define gq_scale_t float
#define GGML_FP32_TO_GQ(x) (x)
#define GGML_GQ_TO_FP32(x) (x)

const int M = 1280;
const int N = 1536;
const int K = 1280;

#define QK 64
#define QB 4
#define gq_t_bits 64
#define gq_quant_t uint64_t

float frand(void) {
    return (float) rand() / (float) RAND_MAX;
}

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

/*static inline int quantize_1_blocks_per_row(int k) {
    return k/QK;
}

static inline int quantize_1_quants_per_block(void) {
    return QK/gq_t_bits;
}

static inline int quantize_1_row_size(int k) {
    const int nb = quantize_1_blocks_per_row(k);
    const int nq = quantize_1_quants_per_block();

    return nb*(2*sizeof(gq_scale_t) + nq*QB*sizeof(gq_quant_t));
}

void quantize_1() {

}*/

void mul_mat_gq1(void* src0, void* src1, float* dst, int m, int n, int k) {
    const int kp = k & ~(gq_t_bits - 1);

    const char * restrict p0 = src0;
    const char * restrict p1 = src1;

    float s0[QB + 1];
    float s1[QB + 1];

    gq_quant_t m0[QB + 1];
    gq_quant_t m1[QB + 1];

    for (int i0 = 0; i0 < m; i0++)
    {
        for (int i1 = 0; i1 < n; i1++)
        {
            float sum = 0.0;
            const char * restrict pp0 = p0 + i0*((2*sizeof(float) + (QK/gq_t_bits)*QB*sizeof(gq_quant_t))*(k/QK));
            const char * restrict pp1 = p1 + i1*((2*sizeof(float) + (QK/gq_t_bits)*QB*sizeof(gq_quant_t))*(k/QK));
            for (int i = 0; i < kp/QK; i++)
            {
                float min0, d0, min1, d1;
                            s0[0] = min0;
                s1[0] = min1;

                for (int b = 0; b < QB; b++) {
                    s0[b + 1] = d0*(1 << b);
                    s1[b + 1] = d1*(1 << b);
                }
            }
        }
    }
}