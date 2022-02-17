#include <cmath>
#include <utility>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <Kokkos_Core.hpp>

struct indexer2d {
  int N;
  indexer2d(int m, int n) : N(n) { }
  int operator()(int x, int y) { return x * N + y; }
};

using matrix_type = Kokkos::View<float**, Kokkos::LayoutRight>;
int jacobi_solver(matrix_type data, int M, int N, float max_diff) {
  matrix_type temp = matrix_type(
      Kokkos::ViewAllocateWithoutInitializing("temp"), M, N);
  Kokkos::deep_copy(temp,data);
  int iterations = 0;
  Kokkos::View<int> keep_going(
      Kokkos::ViewAllocateWithoutInitializing("KeepGoing"));
  matrix_type from = data;
  matrix_type to = temp;
  int h_keep_going;
  do {
    Kokkos::deep_copy(keep_going, 0);
    ++iterations;
    Kokkos::parallel_for(
      Kokkos::MDRangePolicy<
          Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>(
              {1,1}, {M-1,N-1}, {4,64}),
      KOKKOS_LAMBDA (int i, int j) {
        to(i, j) = 0.25f * (from((i-1), j) + from((i+1), j) +
                            from(i , j-1) + from(i, j+1));
        if (!keep_going() &&
            std::fabs(to(i,j) - from(i , j)) > max_diff) {
          keep_going() = 1;
        }
      }
    );
    std::swap(from, to);
    Kokkos::deep_copy(h_keep_going, keep_going);
  } while (h_keep_going > 0);
  if (to == data) {
    Kokkos::deep_copy(data, temp);
  }
  return iterations;
}

int main(int argc, char** argv) {
  Kokkos::initialize(argc,argv);
  {
  int M = argc < 2 ? 1000 : std::atoi(argv[1]);
  int N = argc < 3 ? M : std::atoi(argv[2]);
  indexer2d idx(M, N);
  matrix_type data("data",M,N);
  Kokkos::deep_copy(data,250.f);
  Kokkos::parallel_for("Init1", M, KOKKOS_LAMBDA (const int i) {
    data(i, 0) = 100;
    data(i, N-1) = 400;
  });
  Kokkos::parallel_for("Init2", Kokkos::RangePolicy<>(1,N-1), KOKKOS_LAMBDA (const int j) {
    data(0, j) = 200;
    data(M-1, j) = 300;
  });
/*
  float* mini_data = new float[10 * 10];
  std::fill(mini_data, mini_data + 100, 0.0f);
  mini_data[1] = 100;
  mini_data[8] = 200;
  mini_data[10] = 300;
  jacobi_solver(mini_data, 10, 10, 0.01f);
  delete[] mini_data;
*/
  std::cout << "Solving for a " << M << " x " << N << " matrix.\n";
  auto start = std::chrono::steady_clock::now();

  int iterations = jacobi_solver(data, M, N, 0.01f);

  auto end = std::chrono::steady_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Took " << (duration / 1000) << "." << std::setw(3)
            << std::setfill('0') << (duration % 1000) << "s for "
            << iterations << " iterations.\n";
  }
  Kokkos::finalize();
}
