#include <iostream>
#include <cuda_runtime.h>
using namespace std;
// kernel 1
__global__ void add(const int *A, int *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    C[i] = A[i] + A[i];
}
// kernel 2
__global__ void square(const int *C, int *D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    D[i] = C[i] * C[i];
}
int main()
{
    const int N = 1024;
    const size_t size_bytes = N * sizeof(int);
    // init host
    int *h_A, *h_Result1, *h_Result2;
    h_A = new int[N];
    h_Result1 = new int[N];
    h_Result2 = new int[N];
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = i;
    }
    // init device
    int *d_A, *d_Result1, *d_Result2;
    cudaMalloc((void **)&d_A, size_bytes);
    cudaMalloc((void **)&d_Result1, size_bytes);
    cudaMalloc((void **)&d_Result2, size_bytes);
    cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_Result1, 0, size_bytes);
    cudaMemset(d_Result2, 0, size_bytes);
    // init streams and event
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaEventCreate(&event);
    // config
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // kernel1 stream1
    add<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_Result1);
    cudaEventRecord(event, stream1);
    cudaStreamWaitEvent(stream2, event, 0);
    // kernel2 on stream2
    square<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_Result1, d_Result2);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Result1, d_Result1, size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Result2, d_Result2, size_bytes, cudaMemcpyDeviceToHost);
    return 0;
}
