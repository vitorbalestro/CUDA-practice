#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


#define N // size of the problem data

#define M // number of threads by block

__global__ void add(int *d_a, int *d_b, int *d_c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    d_c[i] = d_a[i] + d_b[i];
}

__host__ int ceil(int A, int B){
    return (A-1)/N + 1;
}

int main() {

    int *d_a, *d_b, *d_c;
    add <<<ceil(N,M), M>>>(d_a,d_b,d_c);

    return 0;

}