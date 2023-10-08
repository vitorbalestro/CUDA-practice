#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>


#define BLOCKS 1000
#define THREADSPERBLOCK 1000
#define size 10

__global__ void incrementVector(int *data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    index = index % size;
    atomicAdd(&data[index],1);
}

int main() {

    int data[1000000];
    int data_out[1000000];
    for(int i = 0; i < 1000000; i++){
        data[i] = i;
    }

    int *d_in;
    cudaMalloc((void**)&d_in, 1000000*sizeof(int));
    cudaMemcpy(d_in, data, 1000000*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); 

    incrementVector<<<1000,1000>>>(d_in);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);


    cudaMemcpy(data_out, d_in, 1000000*sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < 10; i++){
        printf("%d--", data_out[i]);
    }
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\nTotal kernel time: %3.2f ms\n", elapsedTime);



}