#include <stdio.h>
#include <stdlib.h>
#include "/usr/include/postgresql/libpq-fe.h"

struct Record {
    int id;
    char *vec;
};

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
    char *conninfo = "hostaddr=127.0.0.1 user=postgres password=6wk48900 dbname=trabalho-bd-2";
    PGconn *conn = connect_to_db(conninfo);

    PGresult *res;
    struct Record *data;

    res = PQexec(conn, "SELECT * FROM object LIMIT 10");

    if(PQresultStatus(res) != PGRES_TUPLES_OK){
        printf("No data retrieved!\n");
        PQclear(res);
        PQfinish(conn);
        exit(1);
    }

    data = malloc(10*sizeof(struct Record));

    char *vec = PQgetvalue(res,9,1);
    
    printf("%s\n", vec);

    PQclear(res);
    PQfinish(conn);

    return 0;
}