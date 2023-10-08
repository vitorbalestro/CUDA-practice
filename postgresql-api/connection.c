#include <stdio.h>
#include <stdlib.h>
#include "/usr/include/postgresql/libpq-fe.h"


PGconn *connect_to_db(char *conninfo){
    
    PGconn *conn = PQconnectdb(conninfo);

    if(PQstatus(conn) == CONNECTION_BAD){
        printf("Connection to database failed: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        exit(1);
    }

    return conn;
}

int main() {

    PGconn *conn = connect_to_db("hostaddr=127.0.0.1 user=postgres password=6wk48900 dbname=trabalho-bd-2");

    if(PQstatus(conn) == CONNECTION_BAD){
        printf("Connection to database failed: %s\n", PQerrorMessage(conn));
        PQfinish(conn);
        exit(1);
    }

    printf("Database connected!\n");
    printf("User: %s\n", PQuser(conn));
    printf("Database name: %s\n", PQdb(conn));
    printf("Password: %s\n", PQpass(conn));

    PQfinish(conn);
    return 0;
}