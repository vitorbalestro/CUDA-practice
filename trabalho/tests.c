#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

float euclidean_distance(int *vec1, int *vec2, int n){
    int dist = 0;
    for(int i = 0; i < n; i++){
        int diff = vec1[i]-vec2[i];
        dist += diff * diff;
    }
    return sqrt((float) dist);
};


int main() {

    int offset_int = 10000;
    char offset[10];
    snprintf(offset,10,"%d",offset_int);
    char *base_string;
    base_string = (char*)malloc(200*sizeof(char));
    strcpy(base_string,"SELECT * FROM object ORDER BY id OFFSET ");
    strcat(base_string,offset);
    strcat(base_string," LIMIT ");
    char limit[10];
    int limit_int = 20000;
    snprintf(limit,10,"%d",limit_int);
    strcat(base_string,limit);
    printf("%s\n",base_string);

    return 0;

}