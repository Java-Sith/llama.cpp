#include "ggml-xrt.h"

// Function to perform naive matrix multiplication
float** naiveMatrixMultiply(float** matrix1, int rows1, int cols1, float** matrix2, int cols2) {
    // Initialize the result matrix with zeros
    float** result = new float*[rows1];
    for (int i = 0; i < rows1; ++i) {
        result[i] = new float[cols2];
        for (int j = 0; j < cols2; ++j) {
            result[i][j] = 0.0f;
        }
    }

    // Perform matrix multiplication
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return result;
}

