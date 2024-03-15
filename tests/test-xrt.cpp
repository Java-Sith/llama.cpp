#include "ggml-xrt.h"
#define DIM1 1023
#define DIM2 1024
#define DIM3 1025

int main() {
    int m = DIM1, n = DIM2, k = DIM3;
    // Allocate memory for a 2D array of floats
    float** src0 = new float*[m];
    for (int i = 0; i < m; ++i) {
        src0[i] = new float[k];
    }

    // Fill the matrix with some values (optional)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            // Example: filling with row*col as a float
            src0[i][j] = static_cast<float>(i * k + j);
        }
    }

    // Allocate memory for a 2D array of floats
    float** src1 = new float*[n];
    for (int i = 0; i < n; ++i) {
        src1[i] = new float[k];
    }

    // Fill the matrix with some values (optional)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            // Example: filling with row*col as a float
            src1[i][j] = static_cast<float>(i * k + j);
        }
    }
    for (int i = 0; i < 16; ++i) {
        printf("%f %f\n", src0[i], src1[i]);
    }
    double iM = 1.0/m;
    double sumf = 0.0f;
    float** result = naiveMatrixMultiply(src0, m, k, src1, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            sumf += result[i][j];
        }
    }
    std::cout << "PASSED!: Sum = " << sumf << std::endl;
    // Deallocate memory
    for (int i = 0; i < m; ++i) {
        delete[] src0[i];
    }
    delete[] src0;
    // Deallocate memory
    for (int i = 0; i < n; ++i) {
        delete[] src1[i];
    }
    delete[] src1;
    // Deallocate memory
    for (int i = 0; i < m; ++i) {
        delete[] result[i];
    }
    delete[] result;
    return 0;
}