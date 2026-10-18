#include <iostream>
#include <vector>
#include <mpi.h>

// CHUNKED RING ALLGATHER:
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank{0};
    int world_size{0};

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int prev{(rank - 1 + world_size)%world_size};
    const int next{(rank + 1)%world_size};

    constexpr int tag{0};
    constexpr int chunk_size{3};

    // ----------------------------------------------------------
    // 1. local chunk of each rank:
    // ----------------------------------------------------------
    std::vector<int> my_chunk(chunk_size);

    for(int j{0}; j < chunk_size; ++j)
        my_chunk[j] = (rank + 1) * 10 + j;

    // ----------------------------------------------------------
    // 2. buffers that will travel around:
    // ----------------------------------------------------------
    std::vector<int> send_chunk{my_chunk};
    std::vector<int> recv_chunk(chunk_size);
    std::vector<int> gathered(world_size * chunk_size, -1);

    // ----------------------------------------------------------
    // 3. storing chunk in correc pos:
    // ----------------------------------------------------------
    const int init_offset{rank*chunk_size};
    for(int j{0}; j<chunk_size; ++j)
        gathered[init_offset + j] = my_chunk[j];
    
    // ----------------------------------------------------------
    // 4. Ring ALLGATHER:
    // ----------------------------------------------------------
    for(int step{0}; step < world_size-1; ++step)
    {
        MPI_Request requests[2];
        MPI_Irecv(recv_chunk.data(), chunk_size, MPI_INT, prev, tag, MPI_COMM_WORLD, &requests[0]);
        MPI_Isend(send_chunk.data(), chunk_size, MPI_INT, next, tag, MPI_COMM_WORLD, &requests[1]);
        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        const int owner{(rank-step-1+world_size) % world_size};
        const int offset{owner*chunk_size};
        for(int j{0}; j<chunk_size; ++j)
            gathered[offset+j] = recv_chunk[j];

        send_chunk = recv_chunk;
    }
    
    // ----------------------------------------------------------
    // 5. print:
    // ----------------------------------------------------------
    std::cout << "Rank " << rank << ": [";

    for (int i{0}; i < static_cast<int>(gathered.size()); ++i)
    {
        std::cout << gathered[i];

        if (i < static_cast<int>(gathered.size()) - 1)
            std::cout << ", ";
    }

    std::cout << "]\n";

    MPI_Finalize();
    return 0;
}

// // RING TOPOLOGY:
// int main(int argc, char* argv[])
// {
//     MPI_Init(&argc, &argv);

//     int rank{0};
//     int world_size{0};

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     int token_value{0};
//     constexpr int tag{0};

//     const int prev{(rank - 1 + world_size) % world_size};
//     const int next{(rank + 1) % world_size};

//     // rank 0 created a TOKEN:
//     if(rank==0)
//     {
//         MPI_Send(&token_value, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
//         MPI_Recv(&token_value, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         std::cout << "Rank 0 received final token: "
//           << token_value
//           << '\n';
//     }
//     // move clockwise:
//     else
//     {
//         MPI_Recv(&token_value, 1, MPI_INT, prev, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         token_value+=1;
//         MPI_Send(&token_value, 1, MPI_INT, next, tag, MPI_COMM_WORLD);
//     }

//     MPI_Finalize();
//     return 0;
// }

// // CHAIN TOPOLOGY:
// int main(int argc, char* argv[])
// {
//     MPI_Init(&argc, &argv);

//     int rank{0};
//     int world_size{0};

//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);

//     if (world_size < 2)
//     {
//         if (rank == 0)
//         {
//             std::cerr << "Chain experiment requires at least 2 MPI processes.\n";
//         }

//         MPI_Finalize();
//         return 1;
//     }

//     int value{0};
//     constexpr int tag{0};

//     // first rank:
//     if(rank==0)
//     {
        
//         MPI_Send(&value, 1, MPI_INT, rank+1, tag, MPI_COMM_WORLD);
//     }
//     // intermediate ranks +final rank:
//     else
//     {
//         MPI_Recv(&value, 1, MPI_INT, rank-1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         value+=1;

//         // ONLY intermediate ranks:
//         if(rank < world_size - 1)
//             MPI_Send(&value, 1, MPI_INT, rank+1, tag, MPI_COMM_WORLD);
//     }

//     if(rank==world_size-1)
//         std::cout << "Final Value at rank " << rank << " is " << value << "\n";
//     MPI_Finalize();
//     return 0;
// }
