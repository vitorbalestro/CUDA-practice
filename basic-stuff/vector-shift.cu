#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>


__global__ void vector_shift(int *input_vector, int *output_vector, int vector_size){

    __shared__ int array[33];
    int tid = threadIdx.x;
    int globalIdx = tid + blockIdx.x * blockDim.x;

    if(globalIdx < vector_size){
        array[tid] = input_vector[globalIdx];
        __syncthreads();
    }

    if(tid < vector_size-1){
        output_vector[tid] = array[tid+1];
    }
}

int main() {

    int input_vector[33] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33};
    int *output_vector;
    output_vector = (int *)malloc(32 * sizeof(int));
    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, 33 * sizeof(int));
    cudaMalloc((void**)&d_out, 32 * sizeof(int));
    cudaMemcpy(d_in, input_vector, 33*sizeof(int), cudaMemcpyHostToDevice);

    vector_shift<<<1,33>>>(d_in, d_out, 33);

    cudaMemcpy(output_vector, d_out, 32*sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for(int i = 0; i < 32; i++) printf("%d--", output_vector[i]);
    cudaFree(d_in);
    cudaFree(d_out);

}