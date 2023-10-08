#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define N 10 // number of lines
#define M 10 // number of columns

int *random_matrix(int L, int C){
    // L = number of lines;
    // C = number of columns;
    int *output;
    output = (int*)malloc(L*C*sizeof(int));
    for(int i = 0; i < L; i++){
        for(int j = 0; j < C; j++){
            output[j+i*C] = rand() % 1000;
        }
    }
    return output;
}

__global__ void MatAdd(int *d_a, int *d_b, int *d_c, int L, int C){
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    d_c[j+i*C] = d_a[j+i*C] + d_b[j+i*C];
}

int main() {

    int *a, *b, c[N*M];
    
    a = random_matrix(N,M);
    b = random_matrix(N,M);

    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            printf("%d--", a[j+i*M]);
        }
        printf("\n");
    }
    printf("\n");
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            printf("%d--", b[j+i*M]);
        }
        printf("\n");
    }
    printf("\n");
    int *d_a, *d_b, *d_c;

    int size = N*M*sizeof(int);

    cudaMalloc((void**)&d_a,size);
    cudaMalloc((void**)&d_b,size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock (N,M);
    MatAdd<<<1,threadsPerBlock>>>(d_a,d_b,d_c,N,M);

    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    

    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            printf("%d--", c[j+i*M]);
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}