#include <iostream>
#include <mpi.h>
#include <vector>

void printVec(const int rank, const int size, const std::vector<int>&vec)
{
    std::cout << "Rank " << rank << "/" << size << ": ";
    for(const auto v:vec)
        std::cout << v << " ";
    std::cout << '\n';
}

void demo_bcast(int rank, int size)
{
    int x=(rank==0)?42:0;
    std::cout << "Rank " << rank << "/" << size << ": x (before) = " << x << "\n";
    MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::cout << "Rank " << "/" << size << ": x (after) = " << x << "\n";
}

void demo_bcast_vector(int rank, int size)
{
    std::vector<int> y = (rank==0) 
        ? std::vector<int>{10,20,30,40,50} 
        : std::vector<int>{0,0,0,0,0};
    printVec(rank, size, y);
    MPI_Bcast(y.data(), y.size(), MPI_INT, 0, MPI_COMM_WORLD);
    printVec(rank, size, y);
}

void demo_scatter(int rank, int size, int per_rank)
{
    std::vector<int> send_data;
    if(rank==0)
        send_data = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};

    std::vector<int> receive_data(per_rank);
    std::cout << "[Before] Rank " << rank << "/" << size << " ";
    for(const auto v: receive_data)
        std::cout << v << " ";
    std::cout << "\n";

    MPI_Scatter(
        // Send:
        rank==0 ? send_data.data():nullptr,
        per_rank,
        MPI_INT,

        // Receive:
        receive_data.data(),
        per_rank,
        MPI_INT,

        0,
        MPI_COMM_WORLD
    );

    std::cout << "[After] Rank " << rank << "/" << size << " ";
    for(const auto v: receive_data)
        std::cout << v << " ";
    std::cout << "\n";
}

void demo_gather(int rank, int size)
{
    std::vector<int> send_data = {rank*10, rank*10+1};
    std::vector<int> receive_data;

    std::cout << "[Before] Rank " << rank << "/" << size << " ";
    for(const auto v: send_data)
        std::cout << v << " ";
    std::cout << "\n";

    if (rank==0)
        receive_data.resize(2*size);

    MPI_Gather(
        // send:
        send_data.data(),
        2,
        MPI_INT,

        // receive:
        rank==0?receive_data.data() : nullptr,
        2,
        MPI_INT,

        0,
        MPI_COMM_WORLD
    );

    std::cout << "[Before] Rank " << rank << "/" << size << " ";
    for(const auto v: send_data)
        std::cout << v << " ";
    std::cout << "\n";
}

void demo_allgather(int rank, int size)
{
    std::vector<int> send_data = {rank*10, rank*10 + 1};
    std::vector<int> receive_data(2*size);

    std::cout << "[Before] Rank " << rank << "/" << size << " ";
    for(const auto v: send_data)
        std::cout << v << " ";
    std::cout << "\n";

    MPI_Allgather(
        // send:
        send_data.data(),
        2,
        MPI_INT,

        // receive:
        receive_data.data(),
        2,
        MPI_INT,

        MPI_COMM_WORLD
    );

    std::cout << "[Before] Rank " << rank << "/" << size << " ";
    for(const auto v: receive_data)
        std::cout << v << " ";
    std::cout << "\n";
}

void demo_alltoall(int rank, int size)
{
    std::vector<int> send_data(size);
    std::vector<int> receive_data(size);

    // populate the send data:
    for(int destination{0}; destination<size; ++destination)
        send_data[destination] = rank*10 + destination;

    std::cout << "[Before] Rank " << rank << "/" << size << " ";
    for(const auto v: send_data)
        std::cout << v << " ";
    std::cout << "\n";

    MPI_Alltoall(
        // send:
        send_data.data(),
        1,
        MPI_INT,

        // receive:
        receive_data.data(),
        1,
        MPI_INT,

        MPI_COMM_WORLD
    );

    std::cout << "[After] Rank " << rank << "/" << size << " ";
    for(const auto v: receive_data)
        std::cout << v << " ";
    std::cout << "\n";
}

void demo_reduce(int rank, int size)
{
    int local_value = (rank+1)*10;
    int global_sum{0};

    std::cout << "[Before] Rank " << rank << "/" << size << " local value = " << local_value << std::endl;

    MPI_Reduce(
        &local_value,
        &global_sum,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );
    std::cout << "[After] Rank " << rank << "/" << size << " local_value = " << local_value << std::endl;
    
    if(rank==0)
        std::cout << "Global Sum = " << global_sum << std::endl;
}

void demo_allreduce(int rank, int size)
{
    int local_value{(rank+1)*10};
    int global_sum{0};

    std::cout << "[Before] Rank " << rank << "/" << size << " local value = " << local_value << std::endl;

    MPI_Allreduce(
        &local_value,
        &global_sum,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );
    std::cout << "[After] Rank " << rank << "/" << size << " global_sum = " << global_sum << std::endl;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. MBI_Bcast: One rank broadcasts "same data" to every rank
    /*
    MPI_Bcast
    {
        buffer,     // address of data to send/receive
        count,      // number of elements
        datatype,   // rank that is distributing the data

        root,       // rank that is distributing the data
        comm        // communicator
    }
    */
    // demo_bcast_vector(rank, size);

    // 2. MPI_Scatter: one rank distributed "different chunks of data" to different ranks 
    /*
    MPI_Scatter
    (
        sendbuf,    // original array
        sendcount,  // number of elements sent to each rank
        sendtype,   // datatype of sent elements

        recvbuf,    // each rank's receiving buffer
        recvcount,  // number of elements received locally
        recvtype,   // datatype of received elements

        root,       // rank that is distributing the data
        comm        // communicator
    )
    */
    // int per_rank=3;
    // demo_scatter(rank, size, per_rank);

    // 3. MPI_Gather:
    /*
    MPI_Gather
    (
        sendbuf,    // each rank's local data
        sendcount,  // num elements sent by each rank
        sendtype,   // datatype of sent elements

        recvbuf,    // destination buffer on the "root"
        recvcount,  // number of elements received per rank
        recvtype,   // datatype of received elements

        root,       // rank that is distributing the data
        comm        // communicator
    );
    */
    // demo_gather(rank, size);

    // 4. MPI_Allgather: takes buffer from every rank and gives everyone the union:
    /*
    MPI_Allgather
    (
        sendbuf,
        sendcount,
        sendtype,

        recvbuf,
        recvcount,
        recvtype,

        comm        // communicator
    );
    */
    // demo_allgather(rank, size);   

    // 5. MPI_Alltoall:
    /*
    MPI_Alltoall
    (
        sendbuf,    // data each rank wants to distribute
        sendcount,  // elements sent to each destination
        sendtype,   // datatype of sent elements

        recvbuf,    // buffer for incoming data
        recvcount,  // elements received from each source
        recvtype,   // datatype of received elements

        comm        // communicator
    );
    */
    // demo_alltoall(rank, size);

    // 6. MPI_Reduce: like MPI_Gather equipped with an operation
    /*
    MPI_Reduce
    (
        sendbuf,    // each rank's local data

        recvbuf,    // destination buffer on the "root"

        count,      // number of elements per rank

        op      ,   // reduction operation (MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN, MPI_LAND, MPI_LOR)

        root,       // rank that recieves the answer
        comm        // communicator
    );
    */
    // demo_reduce(rank, size);

    // 7. MPI_Allreduce: like MPI_Reduce but every rank receives the results instead of just the root rank
    /*
    MPI_Reduce
    (
        sendbuf,    // each rank's local data

        recvbuf,    // destination buffer on the "root"

        count,      // number of elements per rank

        op,         // reduction operation (MPI_SUM, MPI_PROD, MPI_MAX, MPI_MIN, MPI_LAND, MPI_LOR)

        comm        // communicator
    );
    */
    demo_allreduce(rank, size);
   

    MPI_Finalize();

    return 0;
}


void demo_scatterv(int rank, int size);
void demo_gatherv(int rank, int size);