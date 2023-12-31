#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "/usr/include/postgresql/libpq-fe.h"
#include <time.h>


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


__host__ int ceil(int A, int B){
    return (A-1)/B + 1;
}

int *index_bubble_sort(float *vec, int size){
    int *index_vec;
    index_vec = (int *)malloc(size* sizeof(int));
    for(int i = 0; i < size; i++){
        index_vec[i] = i;
    }
    for(int i = 0; i < size-1; i++){
        for(int j = i+1; j < size; j++){
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

__host__ __device__ int parent(int i){
    if(i > 0) return (i-1)/2;
    return -1;
}

__host__ __device__ int left_child(int i){
    return 2*i+1;
}

__host__ __device__ int right_child(int i){
    return 2*i+2;
}   

__host__ __device__ void max_heapfy(float *vec, int *index_vec, int n,int i){
  int left = left_child(i);
    int right = right_child(i);
    int largest = i;
    if(left < n && vec[left] > vec[i]) largest = left;
    if(right < n && vec[right] > vec[largest]) largest = right;
    if(i != largest){
        float aux = vec[i];
        vec[i] = vec[largest];
        vec[largest] = aux;
        int index_aux = index_vec[i];
        index_vec[i] = index_vec[largest];
        index_vec[largest] = index_aux;
        max_heapfy(vec,index_vec,n,largest);
    }
}

__host__ __device__ void build_max_heap(float *vec, int *index_vec, int n){
    int i, last_parent = parent(n-1);
    for(i=last_parent;i>=0;i--) max_heapfy(vec,index_vec,n,i);
   
}

__device__ int *k_smallest_with_heap(float *vec,int n, int _k){
    float *heap;
    heap = (float *)malloc(_k * sizeof(float));
    int *index_vec;
    index_vec = (int *)malloc(_k * sizeof(int));
    for(int i = 0; i < _k; i++){
        index_vec[i] = i;
    }
    int j;
    for(j=0;j<_k;j++) heap[j] = vec[j];
    build_max_heap(heap,index_vec,_k);
    for(j=_k;j<n;j++){
        if(vec[j] < heap[0]){
            heap[0] = vec[j];
            index_vec[0] = j;
            max_heapfy(heap,index_vec,_k,0);
        }
    }
    return index_vec;
}

__global__ void compute_distances(int *d_data, int *d_query, float *d_distances_r, int *d_indexes,int _k){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int block_adj = blockIdx.x * _k;
    
    __shared__ float thread_distances[T];

    if(i < M){
        int vec[N];
        for(int j = 0; j < N; j++){
            vec[j] = d_data[i*N+j];
        }
        float dist = euclidean_distance(d_query,vec,N);
        thread_distances[tid] = dist;
    }
    __syncthreads();

    if(tid == 0){
        int *index_vec = k_smallest_with_heap(thread_distances,T,_k);
        for(int j = 0; j < _k; j++){
            d_indexes[block_adj + j] = index_vec[j] + blockDim.x*blockIdx.x;
            d_distances_r[block_adj+j] = thread_distances[index_vec[j]];
        }
    }
}

__host__ int *simultaneous_k_smallest(float *vec, int *indexes, int n, int _k){
    float *heap;
    heap = (float *)malloc(_k * sizeof(float));
    int *index_vec;
    index_vec = (int *)malloc(_k * sizeof(int));
    for(int i = 0; i < _k; i++){
        index_vec[i] = i;
    }
    int j;
    for(j=0;j<_k;j++) heap[j] = vec[j];
    build_max_heap(heap,index_vec,_k);
    for(j=_k;j<n;j++){
        if(vec[j] < heap[0]){
            heap[0] = vec[j];
            index_vec[0] = j;
            max_heapfy(heap,index_vec,_k,0);
        }
    }
    int *output;
    output = (int*)malloc(_k*sizeof(int));
    for(j = 0; j < _k; j++){
        output[j] = indexes[index_vec[j]];
    }
    return output;
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
    int *d_indexes;
    float *d_distances_r;
    int data_size = M*N*sizeof(int);
    int numBlocks = ceil(M,T);
    float distances_r[k*numBlocks];
    int indexes[k*numBlocks];


    cudaMalloc((void**)&d_data,data_size);
    cudaMalloc((void**)&d_query,N*sizeof(int));
    cudaMalloc((void**)&d_distances_r, k*numBlocks*sizeof(float));
    cudaMalloc((void**)&d_indexes,k*numBlocks*sizeof(int));
    cudaMemcpy(d_data,&data,data_size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_query,&query,N*sizeof(int),cudaMemcpyHostToDevice);

    /*executing the kernel and returning the output to host*/
    compute_distances<<<numBlocks,T>>>(d_data,d_query,d_distances_r,d_indexes,k);
    cudaMemcpy(&distances_r,d_distances_r,k*numBlocks*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(&indexes, d_indexes, k*numBlocks*sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /*deallocating CUDA memory*/
    cudaFree(d_data);
    cudaFree(d_query);
    cudaFree(d_distances_r);
    cudaFree(d_indexes);

    int *output;
    output = simultaneous_k_smallest(distances_r,indexes,k*numBlocks,k);

    for(int i = 0; i < 10; i++){
        printf("%d--", output[i]);
    }

    printf("\n");
    end = clock();

    double duration = ((double)end-start)/CLOCKS_PER_SEC;

    printf("%f\n",duration);
    
    /*closing connection to DB*/
    PQclear(res);
    PQfinish(conn);

    return 0;

}