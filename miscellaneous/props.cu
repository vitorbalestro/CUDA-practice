#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

int main() {

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp,0);
    printf("Maximum threads per block: %d\n", devProp.maxThreadsPerBlock);
    printf("Maximum threads dimension: (%d, %d, %d)\n",devProp.maxThreadsDim[0],devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("Warp size: %d\n", devProp.warpSize);
    printf("Total global memory: %zu bytes\n", devProp.totalGlobalMem);

    return 0;
}   