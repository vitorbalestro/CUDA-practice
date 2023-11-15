#include<stdio.h>
#include<stdlib.h>

int reduce(int *vec, int n){
    int sum = vec[0];
    for(int i = 1; i < n; i++) sum += vec[i];
    return sum;
};

int main(){

    int query[128];
    for(int i = 0; i < 128; i++){
        query[i] = 1;
    }
    int result = reduce(query,128);
    printf("%d\n", result);

}