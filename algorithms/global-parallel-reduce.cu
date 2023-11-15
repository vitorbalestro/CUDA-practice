#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 1024
#define threadsPerBlock 1024


__global__ void globalmem_reduce(int *s_data, int n){
    int index = threadIdx.x + blockDim.x * blockIdx.x;

    for(int stride = n/2; stride > 0; stride >>= 1){
        if(index < stride){
            s_data[index] += s_data[index + stride];
        }
        __syncthreads();
    }

};


int main() {

    int query[1024];
    for(int i = 0; i < 1024; i++){
        query[i] = 1;
    }
    int *d_query;
    int result;
    cudaMalloc((void**)&d_query,N*sizeof(int));
    cudaMemcpy(d_query,&query,N*sizeof(int),cudaMemcpyHostToDevice);
    int numBlocks = N/threadsPerBlock;
    

    globalmem_reduce<<<numBlocks, threadsPerBlock>>>(d_query,N);

    cudaMemcpy(&result,d_query,sizeof(int),cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(d_query);

    printf("%d\n", result);

    return 0;
}