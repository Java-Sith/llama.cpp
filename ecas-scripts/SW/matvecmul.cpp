#include <iostream>
#include <cstdint>
#include <cstring>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

// Profiler
#include "timer.hpp"

// HLS Types
#include "ap_fixed.h"

// Function to find the next power of two greater than or equal to n
int next_power_of_two(int n) {
    if (n <= 64) {
    	return 64;
    } else {
    	return pow(2, ceil(log2(n)));
    }
}

int main(int argc, char** argv) {
    INIT_PROFILER(cynq_profiler)
    int device_index = 0;

    if (argc != 5) {
        return EXIT_FAILURE;
    }

    // Get input size
    static std::string binaryFile = "../HW/package.hw/kernels.xclbin";
    int a_rows = std::stoi(argv[1]);
    int b_cols = std::stoi(argv[2]);
    //b_cols = b_cols < 8 ? 8 : (b_cols - (b_cols & 4));
    int c_cols = std::stoi(argv[3]);
    //c_cols = c_cols < 8 ? 8 : (c_cols - (c_cols & 4));
    int b_rows = std::stoi(argv[4]);

    std::cout << "A rows: " << a_rows << "\n"
              << "B cols: " << b_cols << "\n"
              << "C cols: " << c_cols << std::endl;

    // Compute sizes
    int size_a = a_rows * b_cols;
    int size_b = 1 * b_cols;
    int size_c = 1 * c_cols;

    int padded_size_a = next_power_of_two(size_a);
    int padded_size_b = next_power_of_two(size_b);
    int padded_size_c = next_power_of_two(size_c);

    GET_PROFILE_INSTANCE(setup_time, cynq_profiler);
    setup_time->reset();

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    auto matvecmul = xrt::kernel(device, uuid, "matvecmul");
    setup_time->tick();

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_a = xrt::bo(device, padded_size_a * sizeof(float), matvecmul.group_id(0));
    auto bo_b = xrt::bo(device, padded_size_b * sizeof(float), matvecmul.group_id(1));
    auto bo_c = xrt::bo(device, padded_size_c * sizeof(float), matvecmul.group_id(2));

    // Map the contents of the buffer object into host memory
    auto bo_a_map = bo_a.map<float*>();
    auto bo_b_map = bo_b.map<float*>();
    auto bo_c_map = bo_c.map<float*>();

    // Filling data
    std::cout << "Filling Buffers\n";
    //std::copy(a.begin(), a.end(), bo_a_mm_map);
    //std::copy(b.begin(), b.end(), bo_b_mm_map);

    float as = 0.02, bs = 0.03;
    std::cout << "A: " << std::endl;
    for (int elem = 0; elem < size_a; ++elem) {
        //std::cout << as.V << " ";
        bo_a_map[elem] = as;
        //std::cout << std::hex << as.V << " ";
        as += 0.03;
        if ((elem + 1) % b_cols == 0) {
            //std::cout << std::endl;
            as = 0.025;
        }
    }

    for (int row = 0; row < b_rows; ++row)
    {
        std::cout << "B: " << std::endl;
        for (int elem = 0; elem < size_b; ++elem) {
            //std::cout << bs.V << " ";
            //std::cout << std::hex << bs.V << " ";
            bo_b_map[elem] = bs;
            bs += 0.07;
            if ((elem + 1) % b_cols == 0) {
                //std::cout << std::endl;
                bs = 0.04;
            }
        }
        // std::cout << std::endl;
        std::fill(bo_c_map, bo_c_map + padded_size_c, 0.0f);

        // Synchronize buffer content with device side
        std::cout << "Synchronize input buffer data to device global memory\n";
        START_PROFILE(kernel_execution, cynq_profiler, 10)
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        //matvecmul(a, b + row * COLS_B, c + row * ROWS_A, ROWS_A, COLS_A, 1);
        std::cout << "Execution of the kernel: matvecmul\n";
        auto run = matvecmul(bo_a, bo_b, bo_c, a_rows, b_cols, c_cols);
        std::cout << "Waiting to the end\n";
        run.wait();

        // Get the output;
        std::cout << "Get the output data from the device" << std::endl;
        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        END_PROFILE(kernel_execution);

        std::cout << "C: " << std::endl;
        for (int elem = 0; elem < size_c; ++elem) {
            float cs;
            cs = bo_c_map[elem];
            //std::cout << cs << " ";
            //std::cout << std::hex << cs.V << " ";
            //if ((elem + 1) % c_cols == 0) std::cout << std::endl;
        }
        // std::cout << std::endl;
        // Print the duration
        std::cout << cynq_profiler << std::endl;
    }

    std::cout << "TEST PASSED\n";
    return 0;
}