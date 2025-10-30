#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10000000 // big number

// function
long long expected_sum(long long n) {
    return n * (n + 1) / 2;
}

int main(int argc, char** argv) {
    int rank, size;
    int *A = NULL;
    int *local_A = NULL;
    int local_n;
    long long partial_sum = 0;
    long long total_sum = 0;
    
    // init mpi
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if N is evenly divisible by the number of processes
    if (N % size != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: Vector size N (%d) must be divisible by the number of processes (%d).\n", N, size);
        }
        MPI_Finalize();
        return 1;
    }

    // size of the sub-vector for each process
    local_n = N / size;

    // Task 1: init vetor in root
    A = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        A[i] = i + 1;
    }
    printf("Task 1 complete\n");

    
    // Task 2: data distribution
    local_A = (int *)malloc(local_n * sizeof(int));
    MPI_Scatter(A, local_n, MPI_INT, local_A, local_n, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Task 2: completed: process %d received %d numbers\n", rank, local_n);

    // free A
    if (rank == 0) {
        free(A);
        A = NULL;
    }

    // Task 3: local comp
    for (int i = 0; i < local_n; i++) {
        partial_sum += local_A[i];
    }
    printf("Process %d: partial sum = %lld\n", rank, partial_sum);

    // free sub-vector
    free(local_A);
    local_A = NULL;

    // Task 4: reduce
    MPI_Reduce(&partial_sum, &total_sum, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Task 5: display result
    if (rank == 0) {
        long long expected = expected_sum(N);
        
        printf("\n\n");
        printf("final mpi sum: %lld\n", total_sum);
        printf("expected sum (formula): %lld\n", expected);

        if (total_sum == expected) {
            printf("correct\n");
        } else {
            printf("something wrong happened\n");
        }
    }

    MPI_Finalize();
    
    return 0;
}
