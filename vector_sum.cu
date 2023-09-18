#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void add(int *d_a, int *d_b, int *d_c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    d_c[i] = d_a[i] + d_b[i];
}

int ceil(int N, int M){
    return (N-1)/M + 1;
}

int main() {

    add <<< ceil (N/M), M >>>();

    return 0;

}