#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define M 1024 // number of threads by block
#define N 10245

__global__ void scalar_multiplication(int *d_a, int *d_b, int *d_c, int Q){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < Q){
        d_b[i] = *d_c * d_a[i];
    }
}

__host__ int ceil(int A, int B) {
    return (A-1)/B + 1;
}

int main() {

    int c = 10;
    int a[N], b[N];
    int *d_a, *d_b;
    int *d_c;

    for(int j = 0; j < N; j++){
        a[j] = rand() % 1000;
    }

    int size = N * sizeof(int);


    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void **)&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);

    scalar_multiplication<<<ceil(N,M),M>>>(d_a,d_b,d_c,N);

    cudaMemcpy(&b, d_b, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    for(int i = 0; i < N; i++){
        printf("%d ", b[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    return 0;
}