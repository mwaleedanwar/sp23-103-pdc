#include <iostream>
#include <cuda_runtime.h>
int main()
{
    const int N = 1024;
    size_t size_bytes = N * sizeof(int);
    // host
    int *h_A = nullptr;
    int *h_B = nullptr;
    h_A = new int[N];
    h_B = new int[N];
    // init A & B
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // device
    int *d_A = nullptr;
    int *d_B = nullptr;
    cudaMalloc((void **)&d_A, size_bytes);
    cudaMalloc((void **)&d_B, size_bytes);
    cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_bytes, cudaMemcpyHostToDevice);

    return 0;
}
