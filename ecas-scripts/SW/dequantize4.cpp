#include <iostream>
#include <cstdint>
#include <cstring>
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"
#include "../HW/dequantize4.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_size> <xclbin_file>" << std::endl;
        return EXIT_FAILURE;
    }

    int k = std::stoi(argv[1]);  // Size of the input (multiple of QK_K)
    std::string binaryFile = argv[2];  // Path to the xclbin file

    // Validate input size
    if (k % QK_K != 0) {
        std::cerr << "Error: Input size must be a multiple of " << QK_K << std::endl;
        return EXIT_FAILURE;
    }

    // Open the Xilinx device
    std::cout << "Opening the device..." << std::endl;
    auto device = xrt::device(0);

    // Load the xclbin file
    std::cout << "Loading the xclbin file: " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    // Create kernel handle
    auto kernel = xrt::kernel(device, uuid, "dequantize_row_q4_K");

    // Allocate Buffers in Global Memory for block_q4_K and output
    std::cout << "Allocating input/output buffers..." << std::endl;
    auto bo_in = xrt::bo(device, k * sizeof(block_q4_K), kernel.group_id(0));  // Input buffer
    auto bo_out = xrt::bo(device, k * sizeof(float), kernel.group_id(1));      // Output buffer

    // Map the input buffer to host memory
    auto bo_in_map = bo_in.map<block_q4_K*>();
    auto bo_out_map = bo_out.map<float*>();

    // Fill the input buffer with dummy quantized data
    std::cout << "Filling input buffer with dummy data..." << std::endl;
    for (int i = 0; i < k / QK_K; ++i) {
        // Initialize the block_q4_K struct (dummy data)
        bo_in_map[i].d = 0x3C00;    // Half-precision scale (1.0f)
        bo_in_map[i].dmin = 0x3800; // Half-precision min (-0.5f)
        for (int j = 0; j < QK_K / 2; ++j) {
            bo_in_map[i].qs[j] = 0xF0;  // Fill with dummy quantized values (4-bit)
        }
    }

    // Synchronize input buffer to device memory
    std::cout << "Syncing input buffer to device..." << std::endl;
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute the kernel
    std::cout << "Running kernel..." << std::endl;
    auto run = kernel(bo_in, bo_out, k);  // Kernel invocation
    run.wait();

    // Synchronize output buffer from device to host
    std::cout << "Syncing output buffer from device..." << std::endl;
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Print output values
    std::cout << "Dequantized values:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << bo_out_map[i] << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl;  // Print in rows for readability
    }

    std::cout << "Test completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}
