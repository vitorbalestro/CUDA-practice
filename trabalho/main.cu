#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "distances.h"
#include "util.h"

#define M 3    // dataset size
#define N 2      // dimension of each vector
#define T 1024     // number of threads per block

__device__ float euclidean_distance(int *vec1, int *vec2, int n){
    int dist = 0;
    for(int i = 0; i < n; i++){
        int diff = vec1[i]-vec2[i];
        dist += diff * diff;
    }
    return sqrt((float) dist);
};

__global__ void compute_distances(int *d_data, int *d_query, float *d_distances_list){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < M){
        int vec[N];
        int j;
        for(j = 0; j < N; j++){
            vec[j] = d_data[i+j];
        }
        d_distances_list[i] = euclidean_distance(d_query,vec,N);
    }
}

int main(){

    int data[M*N] = {1,2,3,4,5,6};     // dataset loaded in main memory -- this might not be possible! 
    int query[N] = {0,0};      // query vector

    // we define the variable below as an array of floats or ints, depending on the considered distance
    float distances_list[M];  // distance_list[i] will contain the distance from data[i] to query

    int num_blocks = ceil(M,T);    // number of blocks given that we have a data set of M vectors and T threads per block

    int *d_query;
    int *d_data;
    float *d_distances_list;

    int query_size = N * sizeof(int);
    int data_size = M * N * sizeof(int);
    int distances_list_size = M * sizeof(float);

    cudaMalloc((void**)&d_query, query_size);
    cudaMalloc((void**)&d_data, data_size);
    cudaMalloc((void**)&d_distances_list, distances_list_size);

    cudaMemcpy(d_query, &query, query_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, &data, data_size, cudaMemcpyHostToDevice);

    compute_distances<<<num_blocks, T>>>(d_data,d_query,d_distances_list);

    cudaMemcpy(&distances_list, d_distances_list, distances_list_size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    cudaFree(d_query);
    cudaFree(d_data);
    cudaFree(d_distances_list);

    for (int i = 0; i < M; i++){
        printf("%f--", distances_list[i]);
    }

    return 0;
}