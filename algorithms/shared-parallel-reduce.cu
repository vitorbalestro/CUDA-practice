#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define threadsPerBlock 128
#define N 1024

__global__ void shared_reduce(int *s_data, int *blocks_vec){
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int local_vec[threadsPerBlock];

    local_vec[tid] = s_data[index];
    __syncthreads();
    
    for(int stride = threadsPerBlock/2; stride > 0; stride >>= 1){
        if(tid < stride){
            local_vec[tid] += local_vec[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0){
        blocks_vec[blockIdx.x] = local_vec[0];
        __syncthreads();
    }

};


int main() {

    int query[1024];
    for(int i = 0; i < 1024; i++){
        query[i] = 1;
    }
    int *d_query;
    int *d_blocks_vec;
    int result;
    cudaMalloc((void**)&d_query,N*sizeof(int));
    cudaMemcpy(d_query,&query,N*sizeof(int),cudaMemcpyHostToDevice);
    int numBlocks = N/threadsPerBlock;
    cudaMalloc((void**)&d_blocks_vec,numBlocks*sizeof(int));
    

    shared_reduce<<<numBlocks, threadsPerBlock>>>(d_query,d_blocks_vec);

   
    int blocks_vec[numBlocks];
    cudaMemcpy(&blocks_vec,d_blocks_vec,numBlocks*sizeof(int),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(d_query);
    cudaFree(d_blocks_vec);
    
    result = 0;
    for(int i = 0; i < numBlocks; i++){
        result += blocks_vec[i];
    }

    printf("%d\n", result);

    return 0;
}