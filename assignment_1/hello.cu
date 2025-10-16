#include <stdio.h>

__global__ void helloWorld()
{
    printf(&quot; Hello from the other side % d\n & quot;, threadIdx.x);
}

int main()
{
    helloWorld<<<1, 6>>>();
    cudaDeviceSynchronize();
    return 0;
}