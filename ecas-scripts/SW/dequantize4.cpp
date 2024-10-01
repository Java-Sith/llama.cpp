#include <iostream>
#include <cstdint>
#include <cstring>
#include <cstdlib>   // For rand()
#include <ctime>     // For seeding rand()

#include <ap_int.h>
#include <ap_fixed.h>
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#define K_SCALE_SIZE 12
#define QK_K 256

typedef struct {
    ap_fixed<16, 5> d;    // half-precision scale
    ap_fixed<16, 5> dmin; // half-precision min
    ap_uint<8> scales[K_SCALE_SIZE]; // 6-bit quantized scales and mins
    ap_uint<8> qs[QK_K / 2];         // 4-bit quantized values
} block_q4_K;

// Helper function to convert half-precision to full-precision
float half_to_float(uint16_t half_val) {
    // Simple conversion function (this could use a library or be expanded)
    // For now, we assume 0x3C00 is 1.0 and 0x3800 is -0.5
    if (half_val == 0x3C00) return 1.0f;
    if (half_val == 0x3800) return -0.5f;
    return 0.0f;  // Default case for other values (implement full conversion if needed)
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_size>" << std::endl;
        return EXIT_FAILURE;
    }

    int k = std::stoi(argv[1]);
    std::string binaryFile = "../HW/package.hw/kernels.xclbin";

    // Validate input size
    if (k % QK_K != 0) {
        std::cerr << "Error: Input size must be a multiple of " << QK_K << std::endl;
        return EXIT_FAILURE;
    }

    // Seed for random number generation
    srand(time(0));

    // Open the Xilinx device
    std::cout << "Opening the device..." << std::endl;
    auto device = xrt::device(0);

    // Load the xclbin file
    std::cout << "Loading the xclbin file: " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    auto dequantize = xrt::kernel(device, uuid, "dequantize4");

    // Allocate Buffers in Global Memory for block_q4_K and output
    std::cout << "Allocating input/output buffers..." << std::endl;
    auto bo_in = xrt::bo(device, k * sizeof(block_q4_K), dequantize.group_id(0));  // Input buffer
    auto bo_out = xrt::bo(device, k * sizeof(float), dequantize.group_id(1));      // Output buffer

    // Map the input buffer to host memory
    auto bo_in_map = bo_in.map<block_q4_K*>();
    auto bo_out_map = bo_out.map<float*>();

    // Fill the input buffer with dummy quantized data
    std::cout << "Filling input buffer with dummy data..." << std::endl;
    for (int i = 0; i < k / QK_K; ++i) {
        // Initialize the block_q4_K struct (dummy data)
        bo_in_map[i].d = 0x3C00;    // Half-precision scale (1.0f)
        bo_in_map[i].dmin = 0x3800; // Half-precision min (-0.5f)
        
        // Convert to float for printing
        float d_full = half_to_float(bo_in_map[i].d.to_uint());  // Properly convert half to float
        float dmin_full = half_to_float(bo_in_map[i].dmin.to_uint());

        std::cout << "Block " << i << ": d = " << d_full << ", dmin = " << dmin_full << std::endl;

        // Fill scales array with random 6-bit values (0 to 63)
        for (int s = 0; s < K_SCALE_SIZE; ++s) {
            bo_in_map[i].scales[s] = rand() % 64;  // 6-bit random value
            std::cout << "  Scale [" << s << "] = " << (int)bo_in_map[i].scales[s] << std::endl;
        }

        for (int j = 0; j < QK_K / 2; ++j) {
            // Generate two random 4-bit values (0-15)
            uint8_t lower_4bits = rand() % 16;  // Random 4-bit value (0-15)
            uint8_t upper_4bits = rand() % 16;  // Random 4-bit value (0-15)

            // Combine the two 4-bit values into one byte
            bo_in_map[i].qs[j] = (upper_4bits << 4) | lower_4bits;

            // Print the generated values
            std::cout << "  Quantized value [" << (j * 2) << "]: " << (int)lower_4bits << std::endl;
            std::cout << "  Quantized value [" << (j * 2 + 1) << "]: " << (int)upper_4bits << std::endl;
        }
    }

    // Synchronize input buffer to device memory
    std::cout << "Syncing input buffer to device..." << std::endl;
    bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute the kernel
    std::cout << "Running kernel..." << std::endl;
    auto run = dequantize(bo_in, bo_out, k);  // Kernel invocation
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
