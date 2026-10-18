# Distributed Numerical Optimization

C++23 and MPI implementations of distributed numerical algorithms, developed as part of the AMS 530 Parallel Computing course at Stony Brook University.

The repository currently includes MPI point-to-point communication exercises, custom collective operations, and one-dimensional distributed dense matrix multiplication. The longer-term goal is to extend these building blocks toward distributed iterative solvers and large-scale numerical optimization.

## Build

**Requirements:** C++23 compiler, CMake, and an MPI implementation (tested with Open MPI).

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

For distributed matrix multiplication:

```bash
mpirun -np 4 ./build/distributed_matmul_1d 1024 5 2
```

Arguments: `matrix_size repetitions warmups [csv_output]`

For example, to save benchmark results:

```bash
mpirun -np 4 ./build/distributed_matmul_1d 1024 5 2 results/matmul_1d.csv
```

## Results and Documentation

- [Technical report (PDF)](docs/notebook/main.pdf)
- [LaTeX source](docs/notebook/main.tex)
- [Benchmark data](results/matmul_1d.csv)
- [Performance figures](results/figures/)
- [Plotting script](scripts/plot_matmul.py)

All twelve tested matrix-multiplication configurations passed correctness verification. The report contains the algorithm, experimental setup, performance measurements, and scalability analysis.