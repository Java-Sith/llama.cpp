//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <cstdint>
#include <cstring>

#include <fstream>

#include <stdio.h>
#include <stdlib.h>

#include "ggml-xrt.h"
#include "ggml-backend-impl.h"
#include "ggml-quants.h"

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define MATRIX_ROW_PADDING 512
#define UNUSED GGML_UNUSED

static int g_device_count = -1;
static int g_all_xrt_device_count = -1;
static int g_main_device = -1;
static int g_main_device_index = -1;

static bool g_xrt_loaded = false;
using DataT = ap_fixed<32, 8>;

bool ggml_xrt_loaded(void) {
    return g_xrt_loaded;
}

bool ggml_backend_is_xrt(ggml_backend_t backend);

typedef void (*ggml_xrt_func_t)(const struct ggml_compute_params *params, struct ggml_tensor * dst);

static size_t ggml_nbytes_split(const struct ggml_tensor * tensor, int nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

struct xrt_device_id2index {
    int index;
};

static xrt_device_id2index g_xrt_device_id2index[GGML_XRT_MAX_DEVICES] = { {-1} };

inline int ggml_xrt_set_device(const int device) {
    int current_device;

    current_device = device;

    // GGML_SYCL_DEBUG("ggml_sycl_set_device device=%d, current_device=%d\n", device, current_device);
    if (device == current_device) {
        return 0;
    }
}

int get_device_id_by_index(int index){
    int res = 0;
    GGML_ASSERT(res>=0);
    return res;
}

int get_device_index_by_id(int id){
    int res = 0;
    GGML_ASSERT(res>=0);
    return res;
}

GGML_CALL static void ggml_xrt_set_main_device(const int main_device) {
    if (main_device >= g_device_count) {
        fprintf(stderr, "warning: cannot set main_device=%d because there are only %d devices. Using device %d instead.\n",
                main_device, g_device_count, g_main_device);
        return;
    }

    if (g_main_device != main_device && g_device_count > 1) {
        g_main_device = main_device;
        //cudaDeviceProp prop;
        //CUDA_CHECK(cudaGetDeviceProperties(&prop, g_main_device));
        //fprintf(stderr, "%s: using device %d (%s) as main device\n", __func__, g_main_device, prop.name);
    }
}

int get_tensor_dimensions(const struct ggml_tensor* tensor) {
    int dimensions = 0;
    for (int i = 0; i < 4; ++i) {
        if (tensor->ne[i] != 0) {
            dimensions++;
        }
    }
    return dimensions;
}

void save_tensor_info(const char* filename, const struct ggml_tensor* tensor) {
    std::ofstream file(filename);

    // Guardar el tipo de datos
    file << "Tipo de datos: " << tensor->type << "\n";

    // Guardar las dimensiones
    file << "Dimensiones: ";
    for (int i = 0; i < 4; ++i) {
        file << tensor->ne[i] << " ";
    }
    file << "\n";

    // Guardar los datos
    file << "Datos: ";
    for (int i = 0; i < tensor->ne[0]; ++i) {
        file << ggml_get_f32_1d(tensor, i) << " ";
    }
    file << "\n";

    file.close();
}

static void ggml_xrt_dup(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    save_tensor_info("Dup.txt", dst);
    ggml_compute_forward_dup(params, dst);
}

// ggml_compute_forward_add

static void ggml_xrt_add(
    const struct ggml_compute_params *params,
    struct ggml_tensor *dst)
{

    save_tensor_info("Add.txt", dst);
    ggml_compute_forward_add(params, dst);
}

static void ggml_xrt_mul(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    save_tensor_info("Mul.txt", dst);
    ggml_compute_forward_mul(params, dst);
}

// ggml_compute_forward_transpose

static void ggml_xrt_nop(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(dst);
}

// ggml_compute_forward_get_rows

static void ggml_xrt_get_rows(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    save_tensor_info("Get Rows.txt", dst);
    ggml_compute_forward_get_rows(params, dst);

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

static void ggml_xrt_rms_norm(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    save_tensor_info("RMS Norm.txt", dst);

    ggml_compute_forward_rms_norm(params, dst);
}

static void ggml_xrt_rope(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    save_tensor_info("Rope.txt", dst);
    ggml_compute_forward_rope(params, dst);
}

static void ggml_xrt_soft_max(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    save_tensor_info("Softmax.txt", dst);
    ggml_compute_forward_soft_max(params, dst);
}

static void ggml_xrt_mul_mat(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    save_tensor_info("Matmul1.txt", src0);
    save_tensor_info("Matmul2.txt", src1);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int ith = params->ith;
    const int nth = params->nth;

    const enum ggml_type type = src0->type;

    const bool src1_cont = ggml_is_contiguous(src1);

    const int64_t ne_plane = ne01*ne00;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    /*const void * wdata    = params->wdata;
    const size_t row_size = ggml_row_size(GGML_TYPE_F32, ne10);

    const int64_t nr0 = ne01;          // src0 rows
    const int64_t nr1 = ne1*ne12*ne13; // src1 rows

    const int64_t nth0 = nr0 > nr1 ? nth : 1; // parallelize by src0 rows
    const int64_t nth1 = nr0 > nr1 ? 1 : nth; // parallelize by src1 rows

    const int64_t ith0 = ith % nth0;
    const int64_t ith1 = ith / nth0;

    const int64_t dr0 = (nr0 + nth0 - 1)/nth0;
    const int64_t dr1 = (nr1 + nth1 - 1)/nth1;

    const int64_t ir010 = dr0*ith0;
    const int64_t ir011 = MIN(ir010 + dr0, nr0);

    const int64_t ir110 = dr1*ith1;
    const int64_t ir111 = MIN(ir110 + dr1, nr1);

    if (ir010 >= ir011 || ir110 >= ir111) {
        sched_yield();
        return;
    }

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int64_t blck_0 = 16;
    const int64_t blck_1 = 16;

    int64_t nrc = 1;
    if ((nr0 % 2 != 0) || (ne11 % 2 != 0)) {
        nrc = 1;
    }

    const size_t src1_col_stride = row_size;*/

    // Compute sizes
    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto matmul = xrt::kernel(device, uuid, "matmul");

    if (params->type == GGML_TASK_FINALIZE) {
        return;
    }

    // perform sgemm, parallelization controlled by blas lib
    if (ith != 0) {
        return;
    }

    //const int64_t tgemm0 = ggml_perf_time_us();
    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const void  * x = (char *) src0->data + i02*nb02 + i03*nb03;
            const float * y = (float *) ((char *) src1->data + i12*nb12 + i13*nb13);
            float * d = (float *) ((char *)  dst->data + i12*nb2  + i13*nb3);

            if (type != GGML_TYPE_F32) {
                x = (float *) params->wdata + i13*ne12*ne_plane + i12*ne_plane;
            }

            auto bo_a_mm = xrt::bo(device, size_a * sizeof(uint32_t), matmul.group_id(0));
            auto bo_b_mm = xrt::bo(device, size_b * sizeof(uint32_t), matmul.group_id(1));
            auto bo_c_mm = xrt::bo(device, size_c * sizeof(uint32_t), matmul.group_id(2));
            auto bo_a_mm_map = bo_a_mm.map<uint32_t*>();
            auto bo_b_mm_map = bo_b_mm.map<uint32_t*>();
            auto bo_c_mm_map = bo_c_mm.map<uint32_t*>();

            /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        ne1, ne01, ne10,
                        1.0f,    y, ne10,
                                x, ne00,
                        0.0f,    d, ne01);*/
        }
    }
}

static void ggml_xrt_unary(
        const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {

    save_tensor_info("Unary.txt", dst);
    ggml_compute_forward_unary(params, dst);
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
        case GGML_OP_GET_ROWS:
            func = ggml_xrt_get_rows;
            break;
        case GGML_OP_ADD:
            func = ggml_xrt_add;
            break;
        case GGML_OP_MUL:
            func = ggml_xrt_mul;
            break;
        case GGML_OP_UNARY:
            func = ggml_xrt_unary;
            break;
        case GGML_OP_SOFT_MAX:
            func = ggml_xrt_soft_max;
            break;
        case GGML_OP_ROPE:
            func = ggml_xrt_rope;
            break;
        case GGML_OP_RMS_NORM:
            func = ggml_xrt_rms_norm;
            break;
        case GGML_OP_MUL_MAT:
            func = ggml_xrt_mul_mat;
            break;
        case GGML_OP_MUL_MAT_ID:
            func = ggml_xrt_mul_mat;
            break;
        case GGML_OP_CPY:
            func = ggml_xrt_dup;
            break;
        case GGML_OP_CONT:
            func = ggml_xrt_dup;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            ggml_xrt_nop(params, tensor);
            break;
        case GGML_OP_DUP:
            //func = ggml_xrt_dup;
            //break;
        case GGML_OP_ACC:
            //func = ggml_xrt_acc;
            //break;
        case GGML_OP_DIV:
            //func = ggml_xrt_div;
            //break;
        case GGML_OP_REPEAT:
            //func = ggml_xrt_repeat;
            //break;
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
        case GGML_OP_DIAG_MASK_INF:
            //func = ggml_xrt_diag_mask_inf;
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
        case GGML_OP_SCALE:
            //func = ggml_xrt_scale;
            //break;
        case GGML_OP_SQR:
            //func = ggml_xrt_sqr;
            //break;
        case GGML_OP_CLAMP:
            //func = ggml_xrt_clamp;
            //break;
        default:
            return false;
    }
    if (params->ith != 0) {
        return true;
    }
    if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
        return true;
    }
    if (tensor->op != GGML_OP_NONE && tensor->op != GGML_OP_RESHAPE && tensor->op != GGML_OP_VIEW &&
    tensor->op != GGML_OP_TRANSPOSE && tensor->op != GGML_OP_PERMUTE)
    {
        func(params, tensor);
    }
    return true;
}

int ggml_xrt_get_device_count() {
    int device_count = 1; //Here comes the device
    return device_count;
}

GGML_CALL void ggml_init_xrt() {
    static bool initialized = false;

    if (!initialized) {

        int user_device_id = get_device_id_by_index(initialized);
        GGML_ASSERT(g_all_xrt_device_count <= GGML_XRT_MAX_DEVICES);
        //ggml_backend_xrt_print_xrt_devices();
        for (int id = 0; id < GGML_XRT_MAX_DEVICES; ++id) {
            g_xrt_device_id2index[id].index = -1;
        }

        int device_inx = -1;

        //hardcode, force set to 1 device
        g_device_count = 1;
        ggml_xrt_set_main_device(user_device_id);
        ggml_xrt_set_device(user_device_id);
        // fprintf(stderr, "Using Device %d\n", user_device_id);

        // for (int id = 0; id < g_all_sycl_device_count; ++id) {
        //     GGML_SYCL_DEBUG("id=%d  g_device_caps[%d].device_id=%d g_sycl_device_id2index[%d].index=%d ", id, id,
        //     g_device_caps[id].device_id, id, g_sycl_device_id2index[id].index);
        // }

        initialized = true;
        g_xrt_loaded = true;
    }
}

////////////////////////////////////////////////////////////////////////////////

// backend interface
struct ggml_backend_xrt_context {
    int device;
    std::string name;
};

struct ggml_backend_xrt_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    //ggml_tensor_extra_gpu * temp_tensor_extras = nullptr;
    //We have one device so must check further
    size_t temp_tensor_extra_index = 0;
    std::string name;

    ggml_backend_xrt_buffer_context(int device, void * dev_ptr) : device(device), dev_ptr(dev_ptr) {}

    ~ ggml_backend_xrt_buffer_context() {

    }

};

GGML_CALL static const char * ggml_backend_xrt_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_xrt_buffer_context * ctx = (ggml_backend_xrt_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_xrt(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_xrt_buffer_get_name;
}

static void
ggml_backend_xrt_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static const size_t TENSOR_ALIGNMENT = 32; // required for mmap as gguf only guarantees 32-byte alignment

static void * ggml_backend_xrt_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_xrt_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                 ggml_tensor *tensor) {
     ggml_backend_xrt_buffer_context * ctx = (ggml_backend_xrt_buffer_context *)buffer->context;

    if (tensor->view_src != NULL && tensor->view_offs == 0) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        tensor->backend = tensor->view_src->backend;
        tensor->extra = tensor->view_src->extra;
        return;
    }

    tensor->backend = GGML_BACKEND_XRT;

    if (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        int64_t row_low = 0;
        int64_t row_high = ggml_nrows(tensor);
        int64_t nrows_split = row_high - row_low;

        size_t original_size = ggml_nbytes_split(tensor, nrows_split);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

    }

    UNUSED(buffer);
}

static void ggml_backend_xrt_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                const void *data, size_t offset,
                                                size_t size) {

    GGML_ASSERT(tensor->backend == GGML_BACKEND_XRT);

    ggml_backend_xrt_buffer_context * ctx = ( ggml_backend_xrt_buffer_context *)buffer->context;

    ggml_xrt_set_device(ctx->device);
    int device_index = get_device_index_by_id(ctx->device);
    memcpy((char *)tensor->data + offset, data, size);

    UNUSED(buffer);
}

static void ggml_backend_xrt_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                const ggml_tensor *tensor,
                                                void *data, size_t offset,
                                                size_t size) {
    GGML_ASSERT(tensor->backend == GGML_BACKEND_XRT);

    ggml_backend_xrt_buffer_context * ctx = ( ggml_backend_xrt_buffer_context *)buffer->context;

    ggml_xrt_set_device(ctx->device);
    int device_index = get_device_index_by_id(ctx->device);
    memcpy(data, (const char *)tensor->data + offset, size);
    UNUSED(buffer);
}

static void ggml_backend_xrt_buffer_clear(ggml_backend_buffer_t buffer,
                                        uint8_t value) {

    ggml_backend_xrt_buffer_context * ctx = ( ggml_backend_xrt_buffer_context *)buffer->context;
    ggml_xrt_set_device(ctx->device);
    int device_index = get_device_index_by_id(ctx->device);
    memset(ctx, value, buffer->size);
}

static struct ggml_backend_buffer_i ggml_backend_xrt_buffer_interface = {
    /* .get_name        = */ ggml_backend_xrt_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_xrt_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_xrt_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_xrt_buffer_init_tensor,
    /* .set_tensor      = */ ggml_backend_xrt_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_xrt_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_xrt_buffer_clear,
    /* .reset           = */ NULL,
};

// sycl buffer type
struct ggml_backend_xrt_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_xrt_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_xrt_buffer_type_context * ctx = (ggml_backend_xrt_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t
ggml_backend_xrt_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) {

    ggml_backend_xrt_buffer_context * buft_ctx = (ggml_backend_xrt_buffer_context *)buft->context;
    int device = (int) buft_ctx->device;

    ggml_xrt_set_device(device);
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0

    void * data = malloc(size); // TODO: use GGML_ALIGNED_MALLOC (move to ggml-impl.h)
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    ggml_backend_xrt_buffer_context * ctx = new ggml_backend_xrt_buffer_context(device, data);

    return ggml_backend_buffer_init(buft, ggml_backend_xrt_buffer_interface, ctx, size);
}

static size_t ggml_backend_xrt_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    UNUSED(buft);
}

static size_t ggml_backend_xrt_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    return SIZE_MAX;

    UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_xrt_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    int64_t row_low = 0;
    int64_t row_high = ggml_nrows(tensor);
    int64_t nrows_split = row_high - row_low;

    size_t size = ggml_nbytes_split(tensor, nrows_split);

    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    UNUSED(buft);
}

static bool ggml_backend_xrt_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_xrt(backend);

    UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_xrt_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_xrt_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_xrt_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_xrt_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_xrt_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_xrt_buffer_type_get_alloc_size,
    /* .supports_backend = */ ggml_backend_xrt_buffer_type_supports_backend,
    /* .is_host          = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_xrt_buffer_type(int device) {
    static struct ggml_backend_buffer_type ggml_backend_xrt_buffer_types[GGML_XRT_MAX_DEVICES];

    static bool ggml_backend_xrt_buffer_type_initialized = false;

    if (!ggml_backend_xrt_buffer_type_initialized) {
        for (int i = 0; i < GGML_XRT_MAX_DEVICES; i++) {
            ggml_backend_xrt_buffer_types[i] = {
                /* .iface    = */ ggml_backend_xrt_buffer_type_interface,
                /* .context  = */ new ggml_backend_xrt_buffer_type_context{i, GGML_XRT_NAME + std::to_string(i)},
            };
        }
        ggml_backend_xrt_buffer_type_initialized = true;
    }

    return &ggml_backend_xrt_buffer_types[device];
}

// host buffer type

GGML_CALL static const char * ggml_backend_xrt_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_XRT_NAME "_Host";

    UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_xrt_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_XRT_NAME "_Host";

    UNUSED(buffer);
}

GGML_CALL void * ggml_xrt_host_malloc(size_t size) {

    void * ptr = nullptr;
    ptr = malloc(size); // TODO: use GGML_ALIGNED_MALLOC (move to ggml-impl.h)
    if (ptr == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ptr;
}

GGML_CALL void ggml_xrt_host_free(void * ptr) {
    free(ptr);
}

static void ggml_backend_xrt_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_xrt_host_free(buffer->context);
}

static ggml_backend_buffer_t ggml_backend_xrt_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_xrt_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_xrt_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_xrt_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_xrt_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_xrt_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_xrt_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // TODO: return device.maxBufferLength
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .supports_backend = */ ggml_backend_cpu_buffer_type()->iface.supports_backend,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_xrt_buffer_type_host;
}

// backend

static const char * ggml_backend_xrt_name(ggml_backend_t backend) {
    return GGML_XRT_NAME;

    UNUSED(backend);
}

static void ggml_backend_xrt_free(ggml_backend_t backend) {
    ggml_backend_xrt_context * xrt_ctx = (ggml_backend_xrt_context *)backend->context;

    delete xrt_ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_xrt_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_xrt_context * xrt_ctx = (ggml_backend_xrt_context *)backend->context;

    return ggml_backend_xrt_buffer_type(xrt_ctx->device);
}

static void ggml_backend_xrt_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               const void *data, size_t offset,
                                               size_t size) {
    ggml_backend_xrt_context * sycl_ctx = (ggml_backend_xrt_context *)backend->context;

    GGML_ASSERT(tensor->buffer->buft == ggml_backend_xrt_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_XRT);
    ggml_backend_tensor_set(tensor, data, offset, size);
}

static void ggml_backend_xrt_get_tensor_async(ggml_backend_t backend,
                                               const ggml_tensor *tensor,
                                               void *data, size_t offset,
                                               size_t size) {
    ggml_backend_xrt_context * xrt_ctx = (ggml_backend_xrt_context *)backend->context;

    GGML_ASSERT(tensor->buffer->buft == ggml_backend_xrt_buffer_type(xrt_ctx->device) && "unsupported buffer type");
    GGML_ASSERT(tensor->backend == GGML_BACKEND_XRT);
    ggml_backend_tensor_get(tensor, data, offset, size);
}

static void ggml_backend_xrt_synchronize(ggml_backend_t backend) {
    ggml_backend_xrt_context * xrt_ctx = (ggml_backend_xrt_context *)backend->context;

    UNUSED(backend);
}

static ggml_backend_graph_plan_t ggml_backend_xrt_graph_plan_create(ggml_backend_t backend, const ggml_cgraph * cgraph) {
    GGML_ASSERT(!"not implemented");

    return nullptr;

    UNUSED(backend);
    UNUSED(cgraph);
}

static void ggml_backend_xrt_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(!"not implemented");

    UNUSED(backend);
    UNUSED(plan);
}

static void ggml_backend_xrt_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(!"not implemented");

    UNUSED(backend);
    UNUSED(plan);
}

static bool ggml_backend_xrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_xrt_context * xrt_ctx = (ggml_backend_xrt_context *)backend->context;

    ggml_xrt_set_main_device(xrt_ctx->device);

    ggml_compute_params params = {};
    params.type = GGML_TASK_COMPUTE;
    params.ith = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE)
            continue;

        assert(node->backend == GGML_BACKEND_XRT);
        assert(node->buffer->buft == ggml_backend_xrt_buffer_type(xrt_ctx->device));
        assert(node->extra != nullptr);

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] != nullptr) {
                assert(node->src[j]->backend == GGML_BACKEND_XRT);
                assert(node->src[j]->buffer->buft == ggml_backend_xrt_buffer_type(xrt_ctx->device));
            }
        }

        bool ok = ggml_xrt_compute_forward(&params, node);
        if (!ok) {
            fprintf(stderr, "%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        GGML_ASSERT(ok);

#if 0
        if (node->type == GGML_TYPE_F32) {
            syclDeviceSynchronize();
            std::vector<float> tmp(ggml_nelements(node), 0.0f);
            syclMemcpy(tmp.data(), node->data, ggml_nelements(node)*sizeof(float), syclMemcpyDeviceToHost);
            printf("\n%s (%s) (%s %s) (%s %s): ", node->name, ggml_op_name(node->op),
                ggml_type_name(node->src[0]->type),
                node->src[1] ? ggml_type_name(node->src[1]->type) : "none",
                node->src[0]->name,
                node->src[1] ? node->src[1]->name : "none");
            double sum = 0.0;
            double sq_sum = 0.0;
            for (int i = 0; i < ggml_nelements(node); i++) {
                printf("%f ", tmp[i]);
                sum += tmp[i];
                sq_sum += tmp[i]*tmp[i];
            }
            printf("\n");
            printf("sum: %f, ", sum);
            printf("sq_sum: %f\n", sq_sum);
        }
#endif
    }

    UNUSED(backend);
    return true;
}

static bool ggml_backend_xrt_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                    return true;
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a;
                struct ggml_tensor * b;
                if (op->op == GGML_OP_MUL_MAT) {
                    a = op->src[0];
                    b = op->src[1];
                } else {
                    a = op->src[2];
                    b = op->src[1];
                }
                if (a->ne[3] != b->ne[3]) {
                    return false;
                }

                if (a->type == GGML_TYPE_IQ1_S) {
                    return false;
                }
                if (a->type == GGML_TYPE_IQ3_XXS) {
                  return false;
                }
                if (a->type == GGML_TYPE_IQ2_XXS) {
                    return false;
                }
                if (a->type == GGML_TYPE_IQ2_XS) {
                    return false;
                }

                return true;
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                if (src0_type == GGML_TYPE_F32) {
                    return true;
                } else {
                    return false;
                }
            } break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_NORM:
        case GGML_OP_REPEAT:
        case GGML_OP_DUP:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ALIBI:
        case GGML_OP_IM2COL:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_LEAKY_RELU:
            return true;
        default:
            return false;
    }

    UNUSED(backend);
}

static ggml_backend_i ggml_backend_xrt_interface = {
    /* .get_name                = */ ggml_backend_xrt_name,
    /* .free                    = */ ggml_backend_xrt_free,
    /* .get_default_buffer_type = */ ggml_backend_xrt_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_xrt_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_xrt_get_tensor_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_xrt_synchronize,
    /* .graph_plan_create       = */ ggml_backend_xrt_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_xrt_graph_plan_free,
    /* .graph_plan_compute      = */ ggml_backend_xrt_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_xrt_graph_compute,
    /* .supports_op             = */ ggml_backend_xrt_supports_op,
};

bool ggml_backend_is_xrt(ggml_backend_t backend) {
    return backend->iface.get_name == ggml_backend_xrt_name;
}

ggml_backend_t ggml_backend_xrt_init(int device) {
    ggml_init_xrt(); // TODO: remove from ggml.c

    ggml_backend_xrt_context * ctx = new ggml_backend_xrt_context {
        /* .device = */ device,
        /* .name   = */ GGML_XRT_NAME + std::to_string(device),
    };

    ggml_backend_t xrt_backend = new ggml_backend {
        /* .interface = */ ggml_backend_xrt_interface,
        /* .context   = */ ctx
    };

    return xrt_backend;
}

static ggml_backend_t ggml_backend_reg_xrt_init(const char * params, void * user_data) {
    ggml_backend_t xrt_backend = ggml_backend_xrt_init((int) (intptr_t) user_data);
    return xrt_backend;

    UNUSED(params);
}

extern "C" int ggml_backend_xrt_reg_devices();

int ggml_backend_xrt_reg_devices() {
    int device_count = ggml_xrt_get_device_count();

    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_XRT_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_xrt_init, ggml_backend_xrt_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}
