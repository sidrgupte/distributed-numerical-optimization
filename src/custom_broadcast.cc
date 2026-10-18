#include <iostream>

#include <vector>

#include <mpi.h>

void linear_broadcast(int& value, int rank, int world_size)
{
    int tag{0};
    if (rank == 0)
    {
        for (int rg{1}; rg < world_size; ++rg)
        {
            MPI_Send(&value, 1, MPI_INT, rg, tag, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(&value, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank{0};
    int world_size{0};

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int value{42};

    // linear broadcast:
    linear_broadcast(value,rank, world_size);

    std::cout << "Rank " << rank << ", Value: " << value << "\n";
    MPI_Finalize();

    return 0;
}