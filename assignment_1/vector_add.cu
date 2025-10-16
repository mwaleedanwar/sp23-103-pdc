#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to perform vector addition on the GPU
__global__ void add_vectors(float *a, float *b, float *c, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n)
    {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    const int N = 10000000;
    vector<float> a_host(N), b_host(N), c_host(N);

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        a_host[i] = static_cast<float>(i);
        b_host[i] = static_cast<float>(i * 2);
    }

    float *a_dev, *b_dev, *c_dev;
    size_t size = N * sizeof(float);

    // Allocate memory on the GPU
    cudaMalloc((void **)&a_dev, size);
    cudaMalloc((void **)&b_dev, size);
    cudaMalloc((void **)&c_dev, size);

    // Copy data from host to device
    cudaMemcpy(a_dev, a_host.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host.data(), size, cudaMemcpyHostToDevice);

    // GPU Timing Start
    auto start_gpu = chrono::high_resolution_clock::now();

    // Configure and launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_vectors<<<blocksPerGrid, threadsPerBlock>>>(a_dev, b_dev, c_dev, N);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // GPU Timing End
    auto end_gpu = chrono::high_resolution_clock::now();

    // Copy result from device to host
    cudaMemcpy(c_host.data(), c_dev, size, cudaMemcpyDeviceToHost);

    chrono::duration<double> diff_gpu = end_gpu - start_gpu;

    cout << "GPU time: " << diff_gpu.count() << " seconds" << endl;

    // Free GPU memory
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    vector<float> a(N), b(N), c(N);

    for (int i = 0; i < N; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    auto start_cpu = chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] + b[i];
    }
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double> diff_cpu = end_cpu - start_cpu;

    cout << "CPU time: " << diff_cpu.count() << " seconds" << endl;

    return 0;
}
