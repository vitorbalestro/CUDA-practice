#include <stdio.h>
#include <stdlib.h>
#include "/usr/include/postgresql/libpq-fe.h"

int main(){

    int lib_ver = PQlibVersion();
    printf("Versão libpq: %d\n", lib_ver);

    return 0;
}