#include <stdio.h>
#include <stdlib.h>

int parent(int i){
    if(i > 0) return (i-1)/2;
    return -1;
}

int left_child(int i){
    return 2*i+1;
}

int right_child(int i){
    return 2*i+2;
}   

void max_heapfy(float *vec, int *index_vec, int n,int i){
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

void build_max_heap(float *vec, int *index_vec, int n){
    int i, last_parent = parent(n-1);
    for(i=last_parent;i>=0;i--) max_heapfy(vec,index_vec,n,i);
   
}

int *k_smallest_with_heap(float *vec,int n, int k){
    float *heap;
    heap = (float *)malloc(k * sizeof(float));
    int *index_vec;
    index_vec = (int *)malloc(k * sizeof(int));
    for(int i = 0; i < k; i++){
        index_vec[i] = i;
    }
    int j;
    for(j=0;j<k;j++) heap[j] = vec[j];
    build_max_heap(heap,index_vec,k);
    for(j=k;j<n;j++){
        if(vec[j] < heap[0]){
            heap[0] = vec[j];
            index_vec[0] = j;
            max_heapfy(heap,index_vec,k,0);
        }
    }
    return index_vec;
}

int *simultaneous_k_smallest(float *vec, int *indexes, int n, int k){
    float *heap;
    heap = (float *)malloc(k * sizeof(float));
    int *index_vec;
    index_vec = (int *)malloc(k * sizeof(int));
    for(int i = 0; i < k; i++){
        index_vec[i] = i;
    }
    int j;
    for(j=0;j<k;j++) heap[j] = vec[j];
    build_max_heap(heap,index_vec,k);
    for(j=k;j<n;j++){
        if(vec[j] < heap[0]){
            heap[0] = vec[j];
            index_vec[0] = j;
            max_heapfy(heap,index_vec,k,0);
        }
    }
    int *output;
    output = (int*)malloc(k*sizeof(int));
    for(j = 0; j < k; j++){
        output[j] = indexes[index_vec[j]];
    }
    return output;
}

int *index_bubble_sort(float *vec, int m){
    int *index_vec;
    index_vec = (int *)malloc(m * sizeof(int));
    for(int i = 0; i < m; i++){
        index_vec[i] = i;
    }
    for(int i = 0; i < m-1; i++){
        for(int j = i+1; j < m; j++){
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

int main() {

    float vec[10] =     {5,6,7,4,3,9,1,2,8,0};
    int index_vec[10] = {1,3,5,4,6,7,8,2,9,0};



    int *output = index_bubble_sort(vec,10);
    for(int i = 0; i < 10; i++) printf("%d--", output[i]);

    return 0;
}