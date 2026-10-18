#include <iostream>

#include <vector>
#include <string>

#include <algorithm>
#include <numeric>

#include <mpi.h>

void print_stats(const std::string&name, std::vector<double> times)
{
    std::sort(times.begin(), times.end());

    const double min_time{times.front()};
    const double max_time{times.back()};
    const double median_time{times[times.size() / 2]};
    const double mean_time{std::accumulate(times.begin(), times.end(), 0.0) / static_cast<double>(times.size())};

    std::cout << name << '\n'
              << "  Min:    " << min_time << " s\n"
              << "  Median: " << median_time << " s\n"
              << "  Mean:   " << mean_time << " s\n"
              << "  Max:    " << max_time << " s\n";
}

void ring_allgather(const std::vector<int>& my_chunk,
                    std::vector<int>&gathered,
                    std::vector<int>& send_chunk,
                    std::vector<int>& recv_chunk,
                    int rank,
                    int world_size)
{
    // ----------------------------------------------------------
    // 1. init:
    // ----------------------------------------------------------
    const int chunk_size{static_cast<int>(my_chunk.size())};

    const int prev{(rank - 1 + world_size)%world_size};
    const int next{(rank + 1)%world_size};

    constexpr int tag{0};

    // ----------------------------------------------------------
    // 2. storing chunk in correc pos:
    // ----------------------------------------------------------
    const int init_offset{rank*chunk_size};
    for(int j{0}; j<chunk_size; ++j)
        gathered[init_offset + j] = my_chunk[j];

    // ----------------------------------------------------------
    // 3. communication:
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
}

// CHUNKED RING ALLGATHER:
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    // ----------------------------------------------------------
    // 1. some init:
    // ----------------------------------------------------------
    int rank{0};
    int world_size{0};

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // ----------------------------------------------------------
    // 2. input chunk size:
    // ----------------------------------------------------------
    int chunk_size{3};
    if(argc>1)
        chunk_size = std::stoi(argv[1]);
    if(chunk_size <= 0)
    {
        if(rank==0)
            std::cerr << "Enter positive chunk size!\n";
        MPI_Finalize();
        return 1;
    }

    // ----------------------------------------------------------
    // 3. local chunk of each rank:
    // ----------------------------------------------------------
    std::vector<int> my_chunk(chunk_size);
    for(int j{0}; j < chunk_size; ++j)
        my_chunk[j] = rank * chunk_size + j;

    // ----------------------------------------------------------
    // 3. buffers that will travel around:
    // ----------------------------------------------------------
    std::vector<int> send_chunk{my_chunk};
    std::vector<int> recv_chunk(chunk_size);
    std::vector<int> ring_gathered(world_size * chunk_size, -1);
    std::vector<int> mpi_gathered(world_size * chunk_size, -1);

    // ----------------------------------------------------------
    // 4A. Ring-ALLGATHER Algorithm warmup runs:
    // ----------------------------------------------------------
    constexpr int warmup_runs{10};
    for (int rep{0}; rep < warmup_runs; ++rep)
    {
        send_chunk = my_chunk;

        MPI_Barrier(MPI_COMM_WORLD);
        ring_allgather(my_chunk, ring_gathered, send_chunk, recv_chunk, rank, world_size);
    }

    // ----------------------------------------------------------
    // 4B. Ring-ALLGATHER Algorithm measured runs:
    // ----------------------------------------------------------
    constexpr int measured_runs{100};

    std::vector<double> ring_times(measured_runs);
    for(int rep{0}; rep < measured_runs; ++rep)
    {
        send_chunk = my_chunk;

        MPI_Barrier(MPI_COMM_WORLD);
        const double start{MPI_Wtime()};

        ring_allgather(my_chunk, ring_gathered, send_chunk, recv_chunk, rank, world_size);
        
        const double end{MPI_Wtime()};

        const double local_time{end - start};
        double global_time{0.0};
        MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank==0)
            ring_times[rep] = global_time;
    }

    // ----------------------------------------------------------
    // 5. print:
    // ----------------------------------------------------------
    // std::cout << "Rank " << rank << ": [";

    // for (int i{0}; i < static_cast<int>(ring_gathered.size()); ++i)
    // {
    //     std::cout << ring_gathered[i];

    //     if (i < static_cast<int>(ring_gathered.size()) - 1)
    //         std::cout << ", ";
    // }

    // std::cout << "]\n";

    // ----------------------------------------------------------
    // 6A: MPI_ALLGATHER: Warmup Runs
    // ----------------------------------------------------------
    for (int rep{0}; rep < warmup_runs; ++rep)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(my_chunk.data(), chunk_size, MPI_INT, mpi_gathered.data(), chunk_size, MPI_INT, MPI_COMM_WORLD);
    }

    // ----------------------------------------------------------
    // 6B: MPI_ALLGATHER: Measured Runs
    // ----------------------------------------------------------
    
    std::vector<double> mpi_times(measured_runs);
    for(int rep{0}; rep < measured_runs; ++rep)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        const double start{MPI_Wtime()};

        MPI_Allgather(my_chunk.data(), chunk_size, MPI_INT, mpi_gathered.data(), chunk_size, MPI_INT, MPI_COMM_WORLD);
    
        const double end{MPI_Wtime()};

        const double local_time{end - start};
        double global_time{0.0};
        MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank==0)
            mpi_times[rep] = global_time;
    }

    // ----------------------------------------------------------
    // 7: compare acc:
    // ----------------------------------------------------------
    const int local_correct{ring_gathered == mpi_gathered ? 1 : 0};
    int global_correct{0};

    MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_LAND, 0,  MPI_COMM_WORLD);
   
    // ----------------------------------------------------------
    // 8: compare time:
    // ----------------------------------------------------------
    

    // ----------------------------------------------------------
    // 9: res:
    // ----------------------------------------------------------
    if (rank == 0)
    {
        std::cout << "Ring Allgather correctness: " << (global_correct ? "PASS" : "FAIL") << '\n';
        print_stats("Ring Times",ring_times);
        print_stats("MPI Times",mpi_times);
    }

    MPI_Finalize();
    return 0;
}