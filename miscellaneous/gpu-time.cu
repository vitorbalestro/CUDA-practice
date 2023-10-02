#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0); 
    // do work...

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total GPU time: %3.1f ms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}