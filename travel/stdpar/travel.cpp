#include <limits>
#include <chrono>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <execution>

constexpr int MAX_CITIES = 15;

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

template <class T,
          class = typename std::enable_if<std::is_integral<T>::value>::type>
inline counting_iterator<T> make_counter(T value) {
  return counting_iterator<T>{value};
}

char const* city_names[MAX_CITIES] = {
  "Atlanta", "Baltimore", "Cleveland", "Denver", "El Paso", "Fort Collins",
  "Green Bay", "Houston", "Indianapolis", "Jacksonville", "Knoxville",
  "Los Angeles", "Memphis", "Nashville", "Orlando"
};

int* init(int N) {
  int* distances = new int[N * N];
  std::mt19937 r;
  std::shuffle(city_names, city_names + N, r);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i == j) {
        distances[i*N + j] = 9999;
      } else if (city_names[i][0] + 1 == city_names[j][0]) {
        distances[i*N + j] = (r() % 15) + 5;
      } else {
        distances[i*N + j] = (r() % 900) + 100;
      }
    }
  }
  return distances;
}

struct route_cost {
  long route;
  int cost;
  route_cost() : route(-1), cost(std::numeric_limits<int>::max()) { }
  route_cost(long route, int cost) : route(route), cost(cost) { }
  static struct min_class {
    route_cost operator()(route_cost const& x, route_cost const& y) const {
      return x.cost < y.cost ? x : y;
    }
  } min;
  static route_cost minf(route_cost const& x, route_cost const& y) {
    return x.cost < y.cost ? x : y;
  }
};
route_cost::min_class route_cost::min;

struct route_iterator {
  long remainder;
  int hops_left;
  unsigned visited = 0;
  route_iterator(long route_id, int num_hops)
    : remainder(route_id), hops_left(num_hops)
  { }
  bool done() const {
    return hops_left <= 0;
  }
  int first() {
    int index = (int)(remainder % hops_left);
    remainder /= hops_left;
    --hops_left;
    visited = (1 << index);
    return index;
  }
  int next() {
    long available = remainder % hops_left;
    remainder /= hops_left;
    --hops_left;
    int index = 0;
    while (true) {
      if ((visited & (1 << index)) == 0) {
        if (--available < 0) {
          break;
        }
      }
      ++index;
    }
    visited |= (1 << index);
    return index;
  }
};

long factorial(long x) {
  if (x <= 1) {
    return 1;
  }
  return x * factorial(x - 1);
}

route_cost find_best_route(int const* distances, int N) {
  return std::transform_reduce(std::execution::par,
      counting_iterator<long>(0L), counting_iterator<long>(factorial(N)),
      route_cost(),
      route_cost::min,
      [=](long i) {
        int cost = 0;
        route_iterator it(i, N);
        int from = it.first();
        while (!it.done()) {
          int to = it.next();
          cost += distances[from*N + to];
          from = to;
        }
        return route_cost(i, cost);
      });
}

void print_route(route_cost best_route, int N) {
  std::cout << "Best route: " << best_route.cost << " miles\n";
  route_iterator it(best_route.route, N);
  std::cout << city_names[it.first()];
  while (!it.done()) {
    std::cout << ", " << city_names[it.next()];
  }
  std::cout << "\n";
}

int main(int argc, char **argv) {
  int N = argc < 2 ? 5 : std::atoi(argv[1]);
  if (N < 1 || N > MAX_CITIES) {
    std::cout << N << " must be between 1 and " << MAX_CITIES << ".\n";
    return 1;
  }
  int const* distances = init(N);

  find_best_route(distances, std::min(N, 5));

  std::cout << "Checking " << factorial(N) 
            << " routes for the best way to visit " << N << " cities...\n";
  auto start = std::chrono::steady_clock::now();

  auto best_route = find_best_route(distances, N);

  auto end = std::chrono::steady_clock::now();
  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Took " << (duration / 1000) << "." << std::setw(3) 
            << std::setfill('0') << (duration % 1000) << "s\n";

  print_route(best_route, N);
}
