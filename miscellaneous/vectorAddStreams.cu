#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define N (1024 * 1024)
#define TOTAL_SIZE (N*21)


__global__ void vectorAdd(int *d_a, int *d_b, int *d_c){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    d_c[i] = d_a[i] + d_b[i];
}

int main() {

    int *h_a, *h_b, *h_c;
    cudaHostAlloc((void**)&h_a,TOTAL_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_b,TOTAL_SIZE * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_c,TOTAL_SIZE * sizeof(int), cudaHostAllocDefault);    
    
    for(int i = 0; i < TOTAL_SIZE; i++){
        h_a[i] = 0;
        h_b[i] = 1;
    }
    int *d_a1, *d_a2, *d_a3;
    int *d_b1, *d_b2, *d_b3;
    int *d_c1, *d_c2, *d_c3;

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    for(int i = 0; i < TOTAL_SIZE; i+=N*3){
        cudaMemcpyAsync(d_a1, h_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_a2, h_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_a3, h_a+i+2*N, N*sizeof(int), cudaMemcpyHostToDevice, stream3);

        cudaMemcpyAsync(d_b1, h_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_b2, h_b+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_b3, h_b+i+2*N, N*sizeof(int), cudaMemcpyHostToDevice, stream3);

        vectorAdd<<<N/256, 256, 0, stream1>>>(d_a1,d_b1,d_c1);
        vectorAdd<<<N/256, 256, 0, stream2>>>(d_a2,d_b2,d_c2);
        vectorAdd<<<N/256, 256, 0, stream3>>>(d_a3,d_b3,d_c3);

        cudaMemcpyAsync(h_c+i, d_c1, N*sizeof(int),cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(h_c+i+N, d_c2, N*sizeof(int),cudaMemcpyDeviceToHost, stream2);
        cudaMemcpyAsync(h_c+i+2*N, d_c3, N*sizeof(int),cudaMemcpyDeviceToHost, stream3);

    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    //frees...

    for(int i = 0; i < 100; i++) {
        printf("%d--", h_c[i]);
    }
    printf("\n");

}