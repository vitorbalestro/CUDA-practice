#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void brickSort(int *d_input, int n) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int aux;
    for(int i = 0; i < n; i++) {
        if(idx > 0 && idx < n){
            if((i % 2 == 1) && (idx % 2 == 0)){
                if(d_input[idx] < d_input[idx-1]){
                    aux = d_input[idx-1];
                    d_input[idx-1] = d_input[idx];
                    d_input[idx] = aux;
                }
                __syncthreads();
            }
            if((i % 2 == 0) && (idx % 2 == 1)){
                if(d_input[idx] < d_input[idx-1]){
                    aux = d_input[idx-1];
                    d_input[idx-1] = d_input[idx];
                    d_input[idx] = aux;
                }
                __syncthreads();
            }
        }
        
    }

}

int main() {

    int N = 19;
    int vec[N] = {0,4,6,5,7,2,3,1,9,8,11,10,12,13,14,17,16,15,18};
    int *d_input;

    int size = N*sizeof(int);

    cudaMalloc((void**)&d_input,size);
    cudaMemcpy(d_input,&vec,size,cudaMemcpyHostToDevice);
    int numBlocks = 1;
    int numThreads = N;

    brickSort<<<numBlocks,numThreads>>>(d_input,N);

    cudaMemcpy(&vec,d_input,size,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < N; i++) {
        printf("%d--", vec[i]);
    }


    return 0;
}