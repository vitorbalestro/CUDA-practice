#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void my_kernel() {
    printf("Hello from GPU!\n");
}

__global__ void add(int *a, int *b, int *c) {
    *c = *a + *b;
}

int main() {
    /*my_kernel<<<3,5>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU!\n");*/

    int a, b, c; // CPU
    int *d_a, *d_b, *d_c; // GPU
    int size = sizeof(int);

    // Allocate space for device
    cudaMalloc((void **)&d_a,size);
    cudaMalloc((void **)&d_b,size);
    cudaMalloc((void**)&d_c,size);

    // Setup input values
    a = 10;
    b = 20;

    // CPU -> GPU
    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

    // kernel execution: 1 thread
    add<<<1,1>>>(d_a,d_b,d_c);
    
    // GPU -> CPU
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Clean memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("%d\n", c);

    return 0;
}