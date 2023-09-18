#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void scalar_multiplication(int *d_a, int *d_b, int *d_c, int N){
    int i = threadIdx.x;
    while(i < N) {
        d_b[i] = *d_c * d_a[i];
        i += blockDim.x;
    }
}

int main() {

    int c = 10;
    int a[5] = {1,2,3,4,5}, b[5];
    int *d_a, *d_b;
    int *d_c;


    int size = 5 * sizeof(int);


    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void **)&d_c, sizeof(int));

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);

    scalar_multiplication<<<1,5>>>(d_a,d_b,d_c,5);

    cudaMemcpy(&b, d_b, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i = 0; i < 5; i++){
        printf("%d ", b[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    


    return 0;
}