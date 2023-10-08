#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "distances.h"
#include "util.h"
#include "/usr/include/postgresql/libpq-fe.h"
#include<time.h>


#define M 10000    // dataset size
#define N 128      // dimension of each vector
#define T 1000     // number of threads per block
#define k 10       // number of desired nearest neighbors

PGconn *connect_to_db(char *conninfo){
    
    PGconn *conn = PQconnectdb(conninfo);

    if(PQstatus(conn) == CONNECTION_BAD){
        printf("Connection to database failed: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        exit(1);
    }

    return conn;
}

int *string_vec2int_vec(char *vec){
    
    int length = strlen(vec);
    
    vec = &vec[1];

    char vec_cpy[length-1];
    strcpy(vec_cpy,vec);
    vec_cpy[length-2] = '\0';

    int *vec_int;
    vec_int = (int*)malloc(128 * sizeof(int));
    int i = 0;
    char *ptr;
    char token[2] = ",";
    
    ptr = strtok(vec_cpy, token);
    
    while(ptr){
        vec_int[i] = atoi(ptr);
        i++;
        ptr = strtok(NULL, token);
    }

    return vec_int;
    
}

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
        for(int j = 0; j < N; j++){
            vec[j] = d_data[i*N+j];
        }
        d_distances_list[i] = euclidean_distance(d_query,vec,N);
    }
}

__host__ int ceil(int A, int B){
    return (A-1)/B + 1;
}

int *index_bubble_sort(float *vec){
    int *index_vec;
    index_vec = (int *)malloc(M * sizeof(int));
    for(int i = 0; i < M; i++){
        index_vec[i] = i;
    }
    for(int i = 0; i < M-1; i++){
        for(int j = i+1; j < M; j++){
            if(vec[j] < vec[i]){
                float temp = vec[j];
                vec[j] = vec[i];
                vec[i] = temp;
                int temp_int = index_vec[j];
                index_vec[j] = index_vec[i];
                index_vec[i] = temp_int;
            }
        }
    }
    return index_vec;

}

int main(){

    clock_t start,end;
    start = clock();

    /*connecting to DB*/
    char *conninfo = "hostaddr=127.0.0.1 user=postgres password=6wk48900 dbname=trabalho-bd-2";
    PGconn *conn = connect_to_db(conninfo);

    /*SQL query*/
    PGresult *res;
    res = PQexec(conn, "SELECT * FROM object ORDER BY id LIMIT 10000");

    /*Preparing the retrieved data*/
    int data[M*N];
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            data[i*N + j] = string_vec2int_vec(PQgetvalue(res,i,1))[j];
        }
    }

    /*declaring variables and copying them to device*/
    int query[N] = {1,3,11,110,62,22,4,0,43,21,22,18,6,28,64,9,11,1,0,0,1,40,101,21,20,2,4,2,2,9,18,35,1,1,7,25,108,116,63,2,0,0,11,74,40,101,116,3,33,1,1,11,14,18,116,116,68,12,5,4,2,2,9,102,17,3,10,18,8,15,67,63,15,0,14,116,80,0,2,22,96,37,28,88,43,1,4,18,116,51,5,11,32,14,8,23,44,17,12,9,0,0,19,37,85,18,16,104,22,6,2,26,12,58,67,82,25,12,2,2,25,18,8,2,19,42,48,11};
    int *d_query;
    int *d_data;
    float *d_distances;
    int data_size = M*N*sizeof(int);
    float distances[M];

    cudaMalloc((void**)&d_data,data_size);
    cudaMalloc((void**)&d_query,N*sizeof(int));
    cudaMalloc((void**)&d_distances, M*sizeof(float));
    cudaMemcpy(d_data,&data,data_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_query,&query,N*sizeof(int),cudaMemcpyHostToDevice);

    /*executing the kernel and returning the output to host*/
    int numBlocks = ceil(M,T);
    compute_distances<<<numBlocks,T>>>(d_data,d_query,d_distances);
    cudaMemcpy(&distances,d_distances,M*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /*deallocating CUDA memory*/
    cudaFree(d_data);
    cudaFree(d_query);
    cudaFree(d_distances);

    int *index_vec;
    index_vec = index_bubble_sort(distances);

    for(int j = 0; j < 10; j++){
        printf("%d--", index_vec[j]);
    }

    end = clock();
    double duration = ((double)end-start)/CLOCKS_PER_SEC;
    printf("\n%f\n",duration);

    /*closing connection to DB*/
    PQclear(res);
    PQfinish(conn);

    return 0;

}