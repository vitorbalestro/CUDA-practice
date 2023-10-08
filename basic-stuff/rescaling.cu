#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define width 1600 // number of lines
#define height 1200 // number of columns
#define type 3 // RGB


__global__ void rescaling(int *d_pin, int *d_pout, int L, int C, int t){
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if(row < t*L && col < t*C){
        if(row >= t*(L/4) && row < 3*t*(L/4) && col >= t*(L/4) && col < 3*t*(L/4))
        d_pout[col + row * t*C] = (d_pout[(col-1) + row * t*C] + d_pout[(col+1) + row * t*C] + d_pout[col + (row-1) * t*C]+ d_pout[col + (row+1) * t*C])/4;

    }

}

int main() {

    int *pin, *pout;
    int *d_pin, *d_pout;

    int size_pin = 3 * width * height * sizeof(char);
    int size_pout = 3 * (width / 2) * (height / 2) * sizeof(char);

    cudaMalloc((void**)&d_pin,size_pin);
    cudaMalloc((void**)&d_pout,size_pout);

    cudaMemcpy(d_pin,&pin,size_pin,cudaMemcpyHostToDevice);

    dim3 threadsPerBlock = (16,16);
    dim3 gridDim = ((height-1)/16 + 1, (width-1)/16 + 1);

    rescaling<<<gridDim,threadsPerBlock>>>(d_pin, d_pout, width, height, type);

    cudaMemcpy(&pout, d_pout, size_pout, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(d_pin);
    cudaFree(d_pout);

    
}