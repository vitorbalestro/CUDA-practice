#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void hillis_steele_scan(int *d_data, int n){

    int index = threadIdx.x + blockDim.x*blockIdx.x;
    
    for(int step = 1; step < n; step = 2*step){
        if(index-step >= 0){
            d_data[index] = d_data[index] + d_data[index-step];
        }
        __syncthreads();
    }

}

int main() {

    int data[6] = {1,3,6,2,9,4};
    int *d_data;
    cudaMalloc((void**)&d_data,6*sizeof(int));
    cudaMemcpy(d_data,&data,6*sizeof(int),cudaMemcpyHostToDevice);
   

    hillis_steele_scan<<<1,6>>>(d_data,6);

    cudaMemcpy(&data,d_data,6*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for(int i = 0; i < 6; i++){
        printf("%d--", data[i]);
    }

    cudaFree(d_data);

    return 0;
}