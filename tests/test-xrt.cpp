#include "ggml-xrt.h"
#define DIM1 1280
#define DIM2 1536
#define DIM3 1280

int main() {
    int m = DIM1, n = DIM2, k = DIM3;
    if (k != m) {
        std::cerr << "Matrix multiplication not possible. Number of columns of first matrix must be equal to number of rows of second matrix." << std::endl;
        return 1;
    }
    // Allocate memory for matrices
    float** matrix1 = allocateMatrix(m, k);
    float** matrix2 = allocateMatrix(k, n);

    // Initialize matrices with random values
    initializeMatrix(matrix1, m, k);
    initializeMatrix(matrix2, k, n);

    // Perform matrix multiplication
    float** resultMatrix = matrixMultiplication(matrix1, matrix2, m, k, n);

    // Print matrices and result
    std::cout << "Matrix 1:" << std::endl;
    printMatrix(matrix1, m, k);
    std::cout << std::endl;

    std::cout << "Matrix 2:" << std::endl;
    printMatrix(matrix2, k, n);
    std::cout << std::endl;

    std::cout << "Resultant Matrix:" << std::endl;
    printMatrix(resultMatrix, m, n);

    // Deallocate memory for matrices
    deallocateMatrix(matrix1, m);
    deallocateMatrix(matrix2, k);
    deallocateMatrix(resultMatrix, m);

    return 0;
}