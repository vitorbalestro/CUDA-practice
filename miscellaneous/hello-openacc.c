#include <stdio.h>
#include "openacc.h"

void hello_world() {
    #pragma acc kernels
    for(int i = 0; i < 5; i++){
        printf("Hello, world!\n");
    }
}