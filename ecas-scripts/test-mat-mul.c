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
                memcpy(&min0, pp0, sizeof(float)); pp0 += sizeof(float);
                memcpy(&d0,   pp0, sizeof(float)); pp0 += sizeof(float);
                memcpy(&min1, pp1, sizeof(float)); pp1 += sizeof(float);
                memcpy(&d1,   pp1, sizeof(float)); pp1 += sizeof(float);
                s0[0] = min0;
                s1[0] = min1;

                for (int b = 0; b < QB; b++) {
                    s0[b + 1] = d0*(1 << b);
                    s1[b + 1] = d1*(1 << b);
                }
                m0[0] = 0-1ULL;
                m1[0] = 0-1ULL;

                for (int s = 0; s < QK/gq_t_bits; ++s) {
                    for (int b = 0; b < QB; b++) {
                        memcpy(&m0[b + 1], pp0, sizeof(gq_quant_t)); pp0 += sizeof(gq_quant_t);
                        memcpy(&m1[b + 1], pp1, sizeof(gq_quant_t)); pp1 += sizeof(gq_quant_t);
                    }

                    for (int q0 = 0; q0 < QB + 1; q0++) {
                        for (int q1 = 0; q1 < QB + 1; q1++) {
                            sum += s0[q0]*s1[q1]*__builtin_popcountll(m0[q0] & m1[q1]);
                        }
                    }
                }
            }
            dst[i0*n + i1] = sum;
        }
    }
}

void mul_mat_gq_2(void * src0, void * src1, float * dst, int m, int n, int k) {
    assert(k % QK == 0);
    for (int i0 = 0; i0 < m; i0++) {
        for (int i1 = 0; i1 < n; i1++) {
            vec_dot_gq_2(k, dst + i1, src0, src1);
            src1 = (const char *) src1 + quantize_2_row_size(k);
        }
        src0 = (const char *) src0 +   quantize_2_row_size(k);
        src1 = (const char *) src1 - n*quantize_2_row_size(k);

        dst = (float *) dst + n;
    }
}

