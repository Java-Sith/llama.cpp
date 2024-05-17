#pragma once

#include "ggml.h"
#include "ggml-impl.h"

#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*#define QK4_0 32
typedef struct {
    ggml_fp16_t d;          // delta
    uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    ggml_fp16_t d;          // delta
    ggml_fp16_t m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_fp16_t) + QK4_1 / 2, "wrong q4_1 block size/padding");*/

//void quantize_row_q4_0(const float * x, block_q4_0 * y, int k);
//void quantize_row_q4_0_reference(const float * x, block_q4_0 * y, int k);
//void dequantize_row_q4_0(const block_q4_0 * x, float * y, int k);
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_XRT_MAX_DEVICES 1
#define GGML_XRT_NAME "XRT"
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

GGML_API void ggml_init_test(void);
GGML_API bool   ggml_test_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor);
GGML_API ggml_backend_t ggml_backend_test_init(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_test_buffer_type(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_test_host_buffer_type(void);
#ifdef  __cplusplus
}
#endif
