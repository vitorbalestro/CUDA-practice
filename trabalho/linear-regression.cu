#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "/usr/include/postgresql/libpq-fe.h"
#include <time.h>
#include <string.h>
#include <sys/time.h>


__global__ void get_coefficients(float *x_vec, float *y_vec, int data_size, float *x_sum, float *x_squared_sum, float *y_sum, float *inner_prod){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < data_size){
        *x_sum += x_vec[idx];
        *x_squared_sum += x_vec[idx] * x_vec[idx];
        *y_sum += y_vec[idx];
        *inner_prod += x_vec[idx] * y_vec[idx];
    }
}

int main() {

    float *x_sum;
    float *x_squared_sum;
    float *y_sum;
    float *inner_prod;

    cudaMallocManaged(&x_sum, sizeof(float));
    cudaMallocManaged(&x_squared_sum, sizeof(float));
    cudaMallocManaged(&y_sum, sizeof(float));
    cudaMallocManaged(&inner_prod,sizeof(float));

    

    return 0;

}