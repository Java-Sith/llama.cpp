#include <iostream>
#include <string>
#include "mat-mul.h"

#if defined(GGML_USE_OPENBLAS)
#include <cblas.h>
#endif

const int M = 1280;
const int N = 1536;
const int K = 1280;

#ifdef GGML_USE_OPENBLAS
void mul_mat_blas(const float * src0, const float * src1, float * dst, int m, int n, int k) {
    assert(k % QK == 0);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, src0, k, src1, n, 0.0f, dst, n);
}
#endif

int main(int argc, char* argv[]) {
    assert(sizeof(gq_quant_t)*8 == gq_t_bits);
    int m = M, n = N, k = K;

    ggml_time_init();

    float * src0  = loadMatrixFromFile(m, k, "ecas-scripts/tensor1.txt");
    float * src1  = loadMatrixFromFile(k, n, "ecas-scripts/tensor2.txt");
    float * dst  = loadMatrix(m, n);

    double iM = 1.0/m;
    double sum = 0.0f;

    int method = 0;
    if (argc > 1) {
        method = std::stoi(argv[1]);
    }

    const int64_t start = ggml_cycles();
    const int64_t start_us = ggml_time_us();

    if (method == 0) {
        #ifdef GGML_USE_OPENBLAS
            mul_mat_blas(src0, src1, dst, m, n, k);
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #else
            mul_mat(src0, src1, dst, m, n, k);
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #endif
    }

    if (method == 1) {
        #ifdef GGML_USE_OPENBLAS
            mul_mat_blas(src0, src1, dst, m, n, k);
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #else
            mul_mat_gq_4(src0, src1, dst, m, n, k);
            saveMatrixToFile((gq_scale_t *) dst, m, n, "ecas-scripts/result.txt");
        #endif
    }
    for (int i = 0; i < n; i++) {
        sum += dst[i]*iM;
    }

    const int64_t end = ggml_cycles();
    const int64_t end_us = ggml_time_us();

    for (int i = 0; i < 16; ++i) {
        printf("%f\n", dst[i]);
    }

    printf("%s: elapsed ticks: %" PRIu64 "\n",  __func__, end - start);
    printf("%s: elapsed us:    %d / %f ms\n",  __func__, (int)(end_us - start_us), (end_us - start_us) / 1000.0);
    printf("%f\n", sum);

    free(src0);
    free(src1);
    free(dst);

    return 0;
}
