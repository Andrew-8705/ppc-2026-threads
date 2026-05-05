#include "kutuzov_i_convex_hull_jarvis/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/partitioner.h>

#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

#include "kutuzov_i_convex_hull_jarvis/common/include/common.hpp"

namespace kutuzov_i_convex_hull_jarvis {

KutuzovITestConvexHullTBB::KutuzovITestConvexHullTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

double KutuzovITestConvexHullTBB::DistanceSquared(double a_x, double a_y, double b_x, double b_y) {
  return ((a_x - b_x) * (a_x - b_x)) + ((a_y - b_y) * (a_y - b_y));
}

double KutuzovITestConvexHullTBB::CrossProduct(double o_x, double o_y, double a_x, double a_y, double b_x, double b_y) {
  return ((a_x - o_x) * (b_y - o_y)) - ((a_y - o_y) * (b_x - o_x));
}

static size_t FindLeftmostPointHelper(const InType &input) {
  struct LeftmostReduction {
    const InType *input_ptr;
    size_t index;
    double x;
    double y;

    explicit LeftmostReduction(const InType &in)
        : input_ptr(&in), index(0), x(std::get<0>((*input_ptr)[0])), y(std::get<1>((*input_ptr)[0])) {}

    LeftmostReduction(LeftmostReduction &other, tbb::split /*unused*/)
        : input_ptr(other.input_ptr), index(other.index), x(other.x), y(other.y) {}

    void operator()(const tbb::blocked_range<size_t> &r) {
      const InType &input = *input_ptr;
      for (size_t i = r.begin(); i != r.end(); ++i) {
        double ix = std::get<0>(input[i]);
        double iy = std::get<1>(input[i]);
        if ((ix < x) || ((ix == x) && (iy < y))) {
          index = i;
          x = ix;
          y = iy;
        }
      }
    }

    void join(const LeftmostReduction &other) {
      if ((other.x < x) || ((other.x == x) && (other.y < y))) {
        index = other.index;
        x = other.x;
        y = other.y;
      }
    }
  };

  LeftmostReduction body(input);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, input.size()), body, tbb::static_partitioner{});
  return body.index;
}

struct NextPointReduction {
  const InType *input_ptr;
  size_t current;
  double cur_x;
  double cur_y;
  size_t next;
  double next_x;
  double next_y;
  double epsilon;

  explicit NextPointReduction(const InType &in, size_t cur, double cx, double cy, double eps)
      : input_ptr(&in),
        current(cur),
        cur_x(cx),
        cur_y(cy),
        next((current + 1) % (*input_ptr).size()),
        next_x(std::get<0>((*input_ptr)[next])),
        next_y(std::get<1>((*input_ptr)[next])),
        epsilon(eps) {}

  NextPointReduction(NextPointReduction &other, tbb::split /*unused*/)
      : input_ptr(other.input_ptr),
        current(other.current),
        cur_x(other.cur_x),
        cur_y(other.cur_y),
        next(other.next),
        next_x(other.next_x),
        next_y(other.next_y),
        epsilon(other.epsilon) {}

  void operator()(const tbb::blocked_range<size_t> &r) {
    const InType &input = *input_ptr;
    for (size_t i = r.begin(); i != r.end(); ++i) {
      if (i == current) {
        continue;
      }

      double i_x = std::get<0>(input[i]);
      double i_y = std::get<1>(input[i]);

      double cross = KutuzovITestConvexHullTBB::CrossProduct(cur_x, cur_y, next_x, next_y, i_x, i_y);

      if (KutuzovITestConvexHullTBB::IsBetterPoint(cross, epsilon, cur_x, cur_y, i_x, i_y, next_x, next_y)) {
        next = i;
        next_x = i_x;
        next_y = i_y;
      }
    }
  }

  void join(const NextPointReduction &other) {
    double cross = KutuzovITestConvexHullTBB::CrossProduct(cur_x, cur_y, next_x, next_y, other.next_x, other.next_y);
    if (KutuzovITestConvexHullTBB::IsBetterPoint(cross, epsilon, cur_x, cur_y, other.next_x, other.next_y, next_x,
                                                 next_y)) {
      next = other.next;
      next_x = other.next_x;
      next_y = other.next_y;
    }
  }
};

static size_t FindNextPointHelper(const InType &input, size_t current, double current_x, double current_y,
                                  double epsilon, size_t &out_next) {
  NextPointReduction body(input, current, current_x, current_y, epsilon);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, input.size()), body, tbb::static_partitioner{});
  out_next = body.next;
  return body.next;
}

bool KutuzovITestConvexHullTBB::IsBetterPoint(double cross, double epsilon, double current_x, double current_y,
                                              double i_x, double i_y, double next_x, double next_y) {
  if (cross < -epsilon) {
    return true;
  }
  if (std::abs(cross) < epsilon) {
    return DistanceSquared(current_x, current_y, i_x, i_y) > DistanceSquared(current_x, current_y, next_x, next_y);
  }
  return false;
}

bool KutuzovITestConvexHullTBB::ValidationImpl() {
  return true;
}

bool KutuzovITestConvexHullTBB::PreProcessingImpl() {
  return true;
}

bool KutuzovITestConvexHullTBB::RunImpl() {
  if (GetInput().size() < 3) {
    GetOutput() = GetInput();
    return true;
  }

  size_t leftmost = FindLeftmostPointHelper(GetInput());
  size_t current = leftmost;
  double current_x = std::get<0>(GetInput()[current]);
  double current_y = std::get<1>(GetInput()[current]);
  const double epsilon = 1e-9;

  while (current != leftmost || GetOutput().empty()) {
    GetOutput().push_back(GetInput()[current]);

    size_t next = 0;
    FindNextPointHelper(GetInput(), current, current_x, current_y, epsilon, next);

    current = next;
    if (current < GetInput().size()) {
      current_x = std::get<0>(GetInput()[current]);
      current_y = std::get<1>(GetInput()[current]);
    }
  }
  return true;
}

bool KutuzovITestConvexHullTBB::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_convex_hull_jarvis
