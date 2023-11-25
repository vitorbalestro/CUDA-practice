#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "/usr/include/postgresql/libpq-fe.h"
#include <time.h>
#include <string.h>

#define TOTAL_SIZE 1000000  // total dataset size
#define M 4000    // size of each stream-step
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

    free(ptr);

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

__global__ void compute_distances(int *d_data, int *d_query, float *d_distances_r, int *d_indexes,int _k, int step){
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
            d_indexes[block_adj + j] = index_vec[j] + blockDim.x*blockIdx.x + step*M;
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
    
    int data[3*M*N];


    
    int query[N] = {1,3,11,110,62,22,4,0,43,21,22,18,6,28,64,9,11,1,0,0,1,40,101,21,20,2,4,2,2,9,18,35,1,1,7,25,108,116,63,2,0,0,11,74,40,101,116,3,33,1,1,11,14,18,116,116,68,12,5,4,2,2,9,102,17,3,10,18,8,15,67,63,15,0,14,116,80,0,2,22,96,37,28,88,43,1,4,18,116,51,5,11,32,14,8,23,44,17,12,9,0,0,19,37,85,18,16,104,22,6,2,26,12,58,67,82,25,12,2,2,25,18,8,2,19,42,48,11};
    int steps = TOTAL_SIZE / (3 * M);
    int numBlocks = ceil(M,T);

    float distances[k*numBlocks*3*steps];
    int indexes[k*numBlocks*3*steps];

    int data_size = M*N*sizeof(int);
    int offset_int;
    int limit_int;


    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);



    /*connecting to DB*/
    char *conninfo = "hostaddr=127.0.0.1 user=postgres password=6wk48900 dbname=trabalho-bd-2";
    PGconn *conn = connect_to_db(conninfo);

    for(int step = 0; step < steps; step++){
        offset_int = 3 * M * step;
        char offset[10];
        snprintf(offset,10,"%d",offset_int);
        limit_int = 3 * M;
        char limit[10];
        snprintf(limit,10,"%d",limit_int);
        char *base_string;
        base_string = (char*)malloc(60*sizeof(char));
        strcpy(base_string,"SELECT * FROM object ORDER BY id OFFSET ");
        strcat(base_string,offset);
        strcat(base_string," LIMIT ");
        strcat(base_string,limit);
        printf("%s\n",base_string);

        PGresult *res;
        res = PQexec(conn, base_string);
        for(int i = 0; i < 3*M; i++){
            int *vec = string_vec2int_vec(PQgetvalue(res,i,1));
            for(int j = 0; j < N; j++){
                
                data[i*N + j] = vec[j];
                
            }
            free(vec);
        }
        PQclear(res);

        int *d_query;
        int *d_data1;
        int *d_data2;
        int *d_data3;
        int *d_indexes1;
        int *d_indexes2;
        int *d_indexes3;
        float *d_distances_r1;
        float *d_distances_r2;
        float *d_distances_r3;

        cudaMalloc((void**)&d_data1,data_size);
        cudaMalloc((void**)&d_data2,data_size);
        cudaMalloc((void**)&d_data3,data_size);

        cudaMalloc((void**)&d_query,N*sizeof(int));
        cudaMalloc((void**)&d_distances_r1, k*numBlocks*sizeof(float));
        cudaMalloc((void**)&d_distances_r2, k*numBlocks*sizeof(float));
        cudaMalloc((void**)&d_distances_r3, k*numBlocks*sizeof(float));

        cudaMalloc((void**)&d_indexes1,k*numBlocks*sizeof(int));
        cudaMalloc((void**)&d_indexes2,k*numBlocks*sizeof(int));
        cudaMalloc((void**)&d_indexes3,k*numBlocks*sizeof(int));

        cudaMemcpy(d_query,&query,N*sizeof(int),cudaMemcpyHostToDevice);

        // each stream copies M vectors to GPU
        cudaMemcpyAsync(d_data1,&data,data_size,cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_data2,&data + data_size,data_size,cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_data3,&data + 2*data_size,data_size,cudaMemcpyHostToDevice, stream3);


        

        /*executing the kernel and returning the output to host*/
        compute_distances<<<numBlocks,T, 0, stream1>>>(d_data1,d_query,d_distances_r1,d_indexes1,k,step);
        compute_distances<<<numBlocks,T, 0, stream2>>>(d_data2,d_query,d_distances_r2,d_indexes2,k,step);
        compute_distances<<<numBlocks,T, 0, stream3>>>(d_data3,d_query,d_distances_r3,d_indexes3,k,step);

        cudaMemcpyAsync(&(distances[step*k*numBlocks]),d_distances_r1,k*numBlocks*sizeof(float),cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(&(distances[(step+1)*k*numBlocks]),d_distances_r2,k*numBlocks*sizeof(float),cudaMemcpyDeviceToHost, stream2);
        cudaMemcpyAsync(&(distances[(step+2)*k*numBlocks]),d_distances_r3,k*numBlocks*sizeof(float),cudaMemcpyDeviceToHost, stream3);
        
        cudaMemcpyAsync(&(indexes[step*k*numBlocks]), d_indexes1, k*numBlocks*sizeof(int),cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(&(indexes[(step+1)*k*numBlocks]), d_indexes2, k*numBlocks*sizeof(int),cudaMemcpyDeviceToHost, stream2);
        cudaMemcpyAsync(&(indexes[(step+2)*k*numBlocks]), d_indexes3, k*numBlocks*sizeof(int),cudaMemcpyDeviceToHost, stream3);

        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);
        cudaStreamSynchronize(stream3);

        /*deallocating CUDA memory*/
        cudaFree(d_data1);
        cudaFree(d_data2);
        cudaFree(d_data3);
        cudaFree(d_query);
        cudaFree(d_distances_r1);
        cudaFree(d_distances_r2);
        cudaFree(d_distances_r3);
        cudaFree(d_indexes1);
        cudaFree(d_indexes2);
        cudaFree(d_indexes3);

        free(base_string);

    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    
    int *output;
    output = simultaneous_k_smallest(distances,indexes,k*numBlocks*steps,k);


    /*adjustment: the DB enumeration starts with 1*/
    for(int i = 0; i < k; i++){
        output[i]++;
    }

    for(int i = 0; i < 10; i++){
        printf("%d--", output[i]);
    }

    printf("\n");
    end = clock();

    double duration = ((double)end-start)/CLOCKS_PER_SEC;

    printf("%f\n",duration);
    
    /*closing connection to DB*/
    PQfinish(conn);

    return 0;

}