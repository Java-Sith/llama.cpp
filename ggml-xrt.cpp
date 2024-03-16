#include "ggml-xrt.h"

// Function to allocate memory for a matrix of size rows x cols
float** allocateMatrix(int rows, int cols) {
    float** matrix = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new float[cols];
    }
    return matrix;
}

// Function to deallocate memory for a matrix
void deallocateMatrix(float** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Function to initialize a matrix with random values
void initializeMatrix(float** matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX; // random float between 0 and 1
        }
    }
}

// Function to perform matrix multiplication
float** matrixMultiplication(float** mat1, float** mat2, int rows1, int cols1, int cols2) {
    float** result = allocateMatrix(rows1, cols2);

    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            result[i][j] = 0.0f;
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}

// Function to print a matrix
void printMatrix(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

