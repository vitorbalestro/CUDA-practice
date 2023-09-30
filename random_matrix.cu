int *random_matrix(int N, int M){
    // N = number of lines;
    // M = number of columns;

    int *output;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            output[j+i*M] = rand() % 1000;
        }
    }
    return output;
}