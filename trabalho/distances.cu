#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "util.h"



__device__ int max_distance(int *vec1, int *vec2, int n){
    int dist = abs(vec1[0] - vec2[0]);
    for(int i = 1; i < n; i++){
        if(abs(vec1[i] - vec2[i]) > dist){
            dist = abs(vec1[i] - vec2[i]);
        }
    }
    return dist;
};

__device__ int l1_distance(int *vec1, int *vec2, int n){
    int dist = 0;
    for(int i = 0; i < n; i++){
        dist += abs(vec1[i] - vec2[i]);
    }
    return dist;
}
