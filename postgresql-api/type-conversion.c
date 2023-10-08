#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int *string_vec2int_vec(char *vec){
    
    int length = strlen(vec);
    
    vec = &vec[1];

    char vec_cpy[length-1];
    strcpy(vec_cpy,vec);
    vec_cpy[length-2] = '\0';
    printf("%s\n",vec_cpy);

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


int main() {

    char *vec = "{21,13,18,11,14,6,4,14,39,54,52,10,8,14,5,2,23,76,65,10,11,23,3,0,6,10,17,5,7,21,20,13,63,7,25,13,4,12,13,112,109,112,63,21,2,1,1,40,25,43,41,98,112,49,7,5,18,57,24,14,62,49,34,29,100,14,3,1,5,14,7,92,112,14,28,5,9,34,79,112,18,15,20,29,75,112,112,50,6,61,45,13,33,112,77,4,18,17,5,3,4,5,4,15,28,4,6,1,7,33,86,71,3,8,5,4,16,72,83,10,5,40,3,0,1,51,36,3}";

    int *vec_int;
    vec_int = string_vec2int_vec(vec);

    for(int j = 0; j < 128 ; j++){
        printf("%d--", vec_int[j]);
    }
    

    return 0;
}