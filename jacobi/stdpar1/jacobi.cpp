#include <cmath>
#include <utility>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <execution>
#include <atomic>

template <class T>
struct counting_iterator {

private:
  typedef counting_iterator<T> self;

public:
  typedef T value_type;
  typedef typename std::make_signed<T>::type difference_type;
  typedef T const* pointer;
  typedef T const& reference;
  typedef std::random_access_iterator_tag iterator_category;

  explicit counting_iterator(value_type v) : value(v) { }

  value_type operator*() const { return value; }
  value_type operator[](difference_type n) const { return value + n; }

  self& operator++() { ++value; return *this; }
  self operator++(int) {
    self result{value};
    ++value;
    return result;
  }
  self& operator--() { --value; return *this; }
  self operator--(int) {
    self result{value};
    --value;
    return result;
  }
  self& operator+=(difference_type n) { value += n; return *this; }
  self& operator-=(difference_type n) { value -= n; return *this; }

  friend self operator+(self const& i, difference_type n) {
    return self(i.value + n);
  }
  friend self operator+(difference_type n, self const& i) {
    return self(i.value + n);
  }
  friend difference_type operator-(self const& x, self const& y) {
    return x.value - y.value;
  }
  friend self operator-(self const& i, difference_type n) {
    return self(i.value - n);
  }

  friend bool operator==(self const& x, self const& y) {
    return x.value == y.value;
  }
  friend bool operator!=(self const& x, self const& y) {
    return x.value != y.value;
  }
  friend bool operator<(self const& x, self const& y) {
    return x.value < y.value;
  }
  friend bool operator<=(self const& x, self const& y) {
    return x.value <= y.value;
  }
  friend bool operator>(self const& x, self const& y) {
    return x.value > y.value;
  }
  friend bool operator>=(self const& x, self const& y) {
    return x.value >= y.value;
  }
private:
  value_type value;
};

struct indexer2d {
  int N;
  indexer2d(int m, int n) : N(n) { }
  int operator()(int x, int y) { return x * N + y; }
};

int jacobi_solver(float* data, int M, int N, float max_diff) {
  float* temp = new float[M * N];
  std::copy(std::execution::par, data, data + M*N, temp);
  int iterations = 0;
  bool keep_going;
  float* from = data;
  float* to = temp;
  do {
    ++iterations;
    keep_going = std::transform_reduce(std::execution::par,
        counting_iterator<int>(N+1), counting_iterator<int>(((M-1)*N)-1),
        false, std::logical_or<bool>(),
        [=](int i) {
          if (i % N != 0 && i % N != N-1) {
            to[i] = 0.25f * (from[i-N] + from[i+N] + from[i-1] + from[i+1]);
            return std::fabs(to[i] - from[i]) > max_diff;
          }
          return false;
        });
    std::swap(from, to);
  } while (keep_going);
  if (to == data) {
    std::copy(std::execution::par, temp, temp + M*N, data);
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
