cmake_minimum_required(VERSION 3.20)

project(distributed_numerical_optimization VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED COMPONENTS CXX)

function(add_mpi_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX)
    target_compile_options(${name} PRIVATE 
        -Wall 
        -Wextra 
        -Wpedantic
    )     
endfunction() 

add_mpi_executable(mpi_comm_lab src/mpi_comm_lab.cc)
add_mpi_executable(custom_ring_allgather src/custom_ring_allgather.cc)
add_mpi_executable(custom_broadcast src/custom_broadcast.cc)
add_mpi_executable(mpi_comm_lab2 src/mpi_collectives_lab.cc)
add_mpi_executable(distributed_matmul_1d src/distributed_matmul_1d.cc)
