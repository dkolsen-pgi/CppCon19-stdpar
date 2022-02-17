#include <cmath>
#include <utility>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

struct indexer2d {
  int N;
  indexer2d(int m, int n) : N(n) { }
  int operator()(int x, int y) { return x * N + y; }
};

int jacobi_solver(float* data, int M, int N, float max_diff) {
  float* temp = new float[M * N];
  int iterations = 0;
  bool keep_going;
  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      temp[i*N + j] = data[i*N + j];
    }
  }
  float* from = data;
  float* to = temp;
  do {
    keep_going = false;
    ++iterations;
    #pragma omp parallel for reduction(||: keep_going)
    for (int i = 1; i < M-1; ++i) {
      for (int j = 1; j < N-1; ++j) {
        to[i*N + j] = 0.25f * (from[(i-1)*N + j] + from[(i+1)*N + j] +
                               from[i*N + j-1] + from[i*N + j+1]);
        if (!keep_going &&
            std::fabs(to[i*N + j] - from[i*N + j]) > max_diff) {
          keep_going = true;
        }
      }
    }
    std::swap(from, to);
  } while (keep_going);
  if (to == data) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        data[i*N + j] = temp[i*N + j];
      }
    }
  }
  delete[] temp;
  return iterations;
}

int main(int argc, char** argv) {
  int M = argc < 2 ? 1000 : std::atoi(argv[1]);
  int N = argc < 3 ? M : std::atoi(argv[2]);
  indexer2d idx(M, N);
  float* data = new float[M * N];
  for (int i = 0; i < M * N; ++i) {
    data[i] = 250;
  }
  for (int i = 0; i < M; ++i) {
    data[idx(i, 0)] = 100;
    data[idx(i, N-1)] = 400;
  }
  for (int j = 1; j < N-1; ++j) {
    data[idx(0, j)] = 200;
    data[idx(M-1, j)] = 300;
  }

  float* mini_data = new float[10 * 10];
  std::fill(mini_data, mini_data + 100, 0.0f);
  mini_data[1] = 100;
  mini_data[8] = 200;
  mini_data[10] = 300;
  jacobi_solver(mini_data, 10, 10, 0.01f);
  delete[] mini_data;

  std::cout << "Solving for a " << M << " x " << N << " matrix.\n";
  auto start = std::chrono::steady_clock::now();

  int iterations = jacobi_solver(data, M, N, 0.01f);


  auto end = std::chrono::steady_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Took " << (duration / 1000) << "." << std::setw(3)
            << std::setfill('0') << (duration % 1000) << "s for "
            << iterations << " iterations.\n";
  delete[] data;
}
