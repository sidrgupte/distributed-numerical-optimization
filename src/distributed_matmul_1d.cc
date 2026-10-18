#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using Matrix = std::vector<double>;

void matmul(const double* A, const double* B, double* C, int rows, int N)
{
    std::fill(C, C + static_cast<std::size_t>(rows) * N, 0.0);

    for (int i = 0; i < rows; ++i)
    {
        for (int k = 0; k < N; ++k)
        {
            const double a = A[static_cast<std::size_t>(i) * N + k];

            for (int j = 0; j < N; ++j)
            {
                C[static_cast<std::size_t>(i) * N + j] += a * B[static_cast<std::size_t>(k) * N + j];
            }
        }
    }
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();

    if (n % 2 == 1)
    {
        return values[n / 2];
    }

    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int P = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2 || argc > 5)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " N [repetitions=5] [warmups=1] [csv_path]\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = 0;
    int repetitions = 5;
    int warmups = 1;

    try
    {
        std::size_t pos = 0;
        N = std::stoi(argv[1], &pos);
        if (pos != std::string{argv[1]}.size())
        {
            throw std::invalid_argument{"Invalid N"};
        }
        if (argc >= 3)
        {
            repetitions = std::stoi(argv[2], &pos);
            if (pos != std::string{argv[2]}.size())
            {
                throw std::invalid_argument{"Invalid repetitions"};
            }
        }
        if (argc >= 4)
        {
            warmups = std::stoi(argv[3], &pos);
            if (pos != std::string{argv[3]}.size())
            {
                throw std::invalid_argument{"Invalid warmups"};
            }
        }
    }
    catch (const std::exception&)
    {
        if (rank == 0)
        {
            std::cerr << "Error: N, repetitions, and warmups must be integers.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (N <= 0 || N % P != 0 || repetitions <= 0 || warmups < 0)
    {
        if (rank == 0)
        {
            std::cerr << "Error: N must be positive and divisible by P; repetitions > 0 and warmups >= 0.\n";
        }
        MPI_Finalize();
        return 1;
    }

    const int local_rows = N / P;
    const std::size_t matrix_size = static_cast<std::size_t>(N) * N;
    const std::size_t local_size = static_cast<std::size_t>(local_rows) * N;

    if (matrix_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        if (rank == 0)
        {
            std::cerr << "Error: matrix exceeds MPI count limit.\n";
        }
        MPI_Finalize();
        return 1;
    }

    Matrix A;
    Matrix B(matrix_size);
    Matrix C;
    Matrix C_serial;
    Matrix A_local(local_size);
    Matrix C_local(local_size);

    if (rank == 0)
    {
        A.resize(matrix_size);
        C.resize(matrix_size);
        C_serial.resize(matrix_size);
        std::mt19937_64 rng{42};
        std::uniform_real_distribution<double> dist{-1.0, 1.0};
        for (auto& value : A)
        {
            value = dist(rng);
        }
        for (auto& value : B)
        {
            value = dist(rng);
        }
    }

    std::vector<double> serial_samples;
    if (rank == 0)
    {
        serial_samples.reserve(static_cast<std::size_t>(repetitions));
        for (int trial = -warmups; trial < repetitions; ++trial)
        {
            const double start = MPI_Wtime();
            matmul(A.data(), B.data(), C_serial.data(), N, N);
            const double elapsed = MPI_Wtime() - start;
            if (trial >= 0)
            {
                serial_samples.push_back(elapsed);
            }
        }
    }

    std::vector<double> parallel_samples;
    std::vector<double> scatter_samples;
    std::vector<double> bcast_samples;
    std::vector<double> compute_samples;
    std::vector<double> gather_samples;

    for (int trial = -warmups; trial < repetitions; ++trial)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        const double parallel_start = MPI_Wtime();

        const double scatter_start = MPI_Wtime();
        MPI_Scatter(rank == 0 ? A.data() : nullptr, static_cast<int>(local_size), MPI_DOUBLE,
                    A_local.data(), static_cast<int>(local_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        const double scatter_time = MPI_Wtime() - scatter_start;

        const double bcast_start = MPI_Wtime();
        MPI_Bcast(B.data(), static_cast<int>(matrix_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        const double bcast_time = MPI_Wtime() - bcast_start;

        const double compute_start = MPI_Wtime();
        matmul(A_local.data(), B.data(), C_local.data(), local_rows, N);
        const double compute_time = MPI_Wtime() - compute_start;

        const double gather_start = MPI_Wtime();
        MPI_Gather(C_local.data(), static_cast<int>(local_size), MPI_DOUBLE,
                   rank == 0 ? C.data() : nullptr, static_cast<int>(local_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        const double gather_time = MPI_Wtime() - gather_start;
        const double parallel_local_time = MPI_Wtime() - parallel_start;

        double local_times[5]{parallel_local_time, scatter_time, bcast_time, compute_time, gather_time};
        double max_times[5]{};
        MPI_Reduce(local_times, max_times, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0 && trial >= 0)
        {
            parallel_samples.push_back(max_times[0]);
            scatter_samples.push_back(max_times[1]);
            bcast_samples.push_back(max_times[2]);
            compute_samples.push_back(max_times[3]);
            gather_samples.push_back(max_times[4]);
        }
    }

    int status = 0;
    if (rank == 0)
    {
        double max_error = 0.0;
        double max_reference = 0.0;
        for (std::size_t i = 0; i < matrix_size; ++i)
        {
            max_error = std::max(max_error, std::abs(C[i] - C_serial[i]));
            max_reference = std::max(max_reference, std::abs(C_serial[i]));
        }

        const double tolerance = 1e-10 * std::max(1.0, max_reference);
        const bool correct = std::isfinite(max_error) && max_error <= tolerance;
        const double serial_time = median(serial_samples);
        const double parallel_time = median(parallel_samples);
        const double speedup = serial_time / parallel_time;
        const double efficiency = speedup / P;
        status = correct ? 0 : 2;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n========================================\n";
        std::cout << " Distributed Matrix Multiplication (1D)\n";
        std::cout << "========================================\n";
        std::cout << "Matrix size          : " << N << " x " << N << '\n';
        std::cout << "MPI processes        : " << P << '\n';
        std::cout << "Rows per process     : " << local_rows << '\n';
        std::cout << "Warmups              : " << warmups << '\n';
        std::cout << "Measured repetitions : " << repetitions << '\n';
        std::cout << "\nMedian Execution Times\n";
        std::cout << "Serial time          : " << serial_time << " s\n";
        std::cout << "Parallel time        : " << parallel_time << " s\n";
        std::cout << "\nMedian Timing Breakdown (rank maxima)\n";
        std::cout << "Scatter              : " << median(scatter_samples) << " s\n";
        std::cout << "Broadcast            : " << median(bcast_samples) << " s\n";
        std::cout << "Local computation    : " << median(compute_samples) << " s\n";
        std::cout << "Gather               : " << median(gather_samples) << " s\n";
        std::cout << "\nPerformance\n";
        std::cout << "Speedup              : " << speedup << "x\n";
        std::cout << "Parallel efficiency  : " << efficiency * 100.0 << "%\n";
        std::cout << "\nCorrectness\n";
        std::cout << "Maximum abs. error   : " << std::scientific << max_error << '\n';
        std::cout << "Tolerance            : " << tolerance << '\n';
        std::cout << "Status               : " << (correct ? "PASS" : "FAIL") << '\n';
        std::cout << "========================================\n";

        if (argc == 5)
        {
            const std::string path{argv[4]};
            std::ifstream existing{path};
            const bool needs_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
            existing.close();
            std::ofstream csv{path, std::ios::app};
            if (!csv)
            {
                std::cerr << "Error: cannot open CSV file: " << path << '\n';
                status = 3;
            }
            else
            {
                if (needs_header)
                {
                    csv << "N,P,warmups,repetitions,serial_s,parallel_s,scatter_s,bcast_s,compute_s,gather_s,speedup,efficiency,max_error,tolerance,correct\n";
                }
                csv << std::setprecision(12) << N << ',' << P << ',' << warmups << ',' << repetitions << ','
                    << serial_time << ',' << parallel_time << ',' << median(scatter_samples) << ','
                    << median(bcast_samples) << ',' << median(compute_samples) << ',' << median(gather_samples) << ','
                    << speedup << ',' << efficiency << ',' << max_error << ',' << tolerance << ','
                    << (correct ? "PASS" : "FAIL") << '\n';
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
