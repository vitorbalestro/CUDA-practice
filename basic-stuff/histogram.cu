#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define N 10 // histogram size

__device__ int detectRange(int input){
    if(input >= 0 && input < 10) return 0;
    if(input >= 10 && input < 20) return 1;
    if(input >= 20 && input < 30) return 2;
    if(input >= 30 && input < 40) return 3;
    if(input >= 40 && input < 50) return 4;
    if(input >= 50 && input < 60) return 5;
    if(input >= 60 && input < 70) return 6;
    if(input >= 70 && input < 80) return 7;
    if(input >= 80 && input < 90) return 8;
    if(input >= 90 && input <= 100) return 9;

    return -1;
    
}

__global__ void createHistogram(int *data, int *histogram, int data_size){

    __shared__ int sub_histogram[N];
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    atomicAdd(&sub_histogram[detectRange(data[tid])],1);
    __syncthreads();

    if(threadIdx.x == 0){
        for(int i = 0; i < N; i++){
            atomicAdd(&histogram[i],sub_histogram[i]);
        }
    }

}