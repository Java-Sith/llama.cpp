/**
* Copyright (C) 2019-2021 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

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

using DataT = ap_fixed<16, 6>;

typedef union {
  uint16_t rvalues[4];
  uint64_t packet;
} PacketU;

int main(int argc, char** argv) {
    //INIT_PROFILER(cynq_profiler)
    int device_index = 0;

    if (argc != 5) {
        return EXIT_FAILURE;
    }
    
    // Get input size
    std::string binaryFile{argv[1]};
    int a_rows = std::stoi(argv[2]);
    int b_cols = std::stoi(argv[3]);
    b_cols = b_cols < 8 ? 8 : (b_cols - (b_cols & 4));
    int c_cols = std::stoi(argv[4]);
    c_cols = c_cols < 8 ? 8 : (c_cols - (c_cols & 4));

    std::cout << "A rows: " << a_rows << "\n"
              << "B cols: " << b_cols << "\n"
              << "C cols: " << c_cols << std::endl;

    // Compute sizes
    int size_a = a_rows * b_cols;
    int size_b = c_cols * b_cols;
    int size_c = a_rows * c_cols;

    //GET_PROFILE_INSTANCE(setup_time, cynq_profiler);
    //setup_time->reset();

    std::cout << "Open the device " << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);;

    auto krnl = xrt::kernel(device, uuid, "matmul");
    setup_time->tick();

    std::cout << "Allocate Buffer in Global Memory\n";
    auto bo_a = xrt::bo(device, size_a * sizeof(uint16_t), krnl.group_id(0));
    auto bo_b = xrt::bo(device, size_b * sizeof(uint16_t), krnl.group_id(1));
    auto bo_c = xrt::bo(device, size_c * sizeof(uint16_t), krnl.group_id(2));

    // Map the contents of the buffer object into host memory
    auto bo_a_map = bo_a.map<uint16_t*>();
    auto bo_b_map = bo_b.map<uint16_t*>();
    auto bo_c_map = bo_c.map<uint16_t*>();
    
    // Filling data
    std::cout << "Filling Buffers\n";
    DataT as = 0.002, bs = 0.003;
    //std::cout << "A: " << std::endl;
    for (int elem = 0; elem < size_a; ++elem) {
        //std::cout << as << " ";
        bo_a_map[elem] = as.V;
        as += DataT{0.003};
        if ((elem + 1) % b_cols == 0) {
            //std::cout << std::endl;
            as = 0.0025;
        }
    }
    //std::cout << "B: " << std::endl;
    for (int elem = 0; elem < size_b; ++elem) {
        //std::cout << bs << " ";
        bo_b_map[elem] = bs.V;
        bs += DataT{0.007};
        if ((elem + 1) % b_cols == 0) {
            //std::cout << std::endl;
            bs = 0.004;
        }
    }
    std::fill(bo_c_map, bo_c_map + size_c, 0);

    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";
    //START_PROFILE(kernel_execution, cynq_profiler, 1000)

    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Start the clock
    auto start_time = std::chrono::high_resolution_clock::now();

    //std::cout << "Execution of the kernel\n";
    auto run = krnl(bo_a, bo_b, bo_c, a_rows, b_cols, c_cols);
    //std::cout << "Waiting to the end\n";
    run.wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto matmul_time = (end_time - start_time)/std::chrono::milliseconds(1);

    // Get the output;
    //std::cout << "Get the output data from the device" << std::endl;
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    //END_PROFILE(kernel_execution);
    /*
    std::cout << "C: " << std::endl;
    for (int elem = 0; elem < size_c; ++elem) {
        DataT cs;
        cs.V = bo_c_map[elem];
        std::cout << cs << " ";
        if ((elem + 1) % c_cols == 0) std::cout << std::endl;
    }*/
    //std::cout << cynq_profiler << std::endl;
    std::cout << "Matrix multiplication = " << matmul_time << " ms " << '\n';
    std::cout << "TEST PASSED\n";
    return 0;
}
