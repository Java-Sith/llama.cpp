#include "ggml-xrt.h"
#define DIM1 1023
#define DIM2 1024
#define DIM3 1025

int main() {
    int m = DIM1, n = DIM2, k = DIM3;
    int size_a = m*k, size_b = n*k, size_c = m*n;
    std::vector<float> ha(size_a);
    std::vector<float> hb(size_b);
    std::vector<float> hc(size_c);
    srand(1);
    for(int i = 0; i < size_a; ++i)
    {
        ha[i] = rand() % 17;
    }
    for(int i = 0; i < size_b; ++i)
    {
        hb[i] = rand() % 17;
    }
    for(int i = 0; i < size_c; ++i)
    {
        hc[i] = rand() % 17;
    }
    mul_mat(ha.data(), hb.data(), hc.data(), m, n, k);
    std::cout << "PASSED!: Sum = " << hc.data() << std::endl;
    return 0;
}