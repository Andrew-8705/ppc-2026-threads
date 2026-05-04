#include "kutuzov_i_convex_hull_jarvis/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/tbb.h>

#include <cmath>
#include <cstddef>
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

size_t KutuzovITestConvexHullTBB::FindLeftmostPoint(const InType &input) {
  struct LeftmostReduction {
    const InType &input;
    size_t index;
    double x, y;

    LeftmostReduction(const InType &in) : input(in), index(0), x(std::get<0>(input[0])), y(std::get<1>(input[0])) {}

    LeftmostReduction(LeftmostReduction &other, tbb::split)
        : input(other.input), index(other.index), x(other.x), y(other.y) {}

    void operator()(const tbb::blocked_range<size_t> &r) {
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

  size_t leftmost = FindLeftmostPoint(GetInput());

  size_t current = leftmost;
  double current_x = std::get<0>(GetInput()[current]);
  double current_y = std::get<1>(GetInput()[current]);

  const double epsilon = 1e-9;

  while (true) {
    GetOutput().push_back(GetInput()[current]);

    struct NextPointReduction {
      const InType &input;
      size_t current;
      double cur_x, cur_y;
      size_t next;
      double next_x, next_y;
      double epsilon;

      NextPointReduction(const InType &in, size_t cur, double cx, double cy, double eps)
          : input(in), current(cur), cur_x(cx), cur_y(cy), epsilon(eps) {
        next = (current + 1) % input.size();
        next_x = std::get<0>(input[next]);
        next_y = std::get<1>(input[next]);
      }

      NextPointReduction(NextPointReduction &other, tbb::split)
          : input(other.input),
            current(other.current),
            cur_x(other.cur_x),
            cur_y(other.cur_y),
            next(other.next),
            next_x(other.next_x),
            next_y(other.next_y),
            epsilon(other.epsilon) {}

      void operator()(const tbb::blocked_range<size_t> &r) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          if (i == current) {
            continue;
          }

          double i_x = std::get<0>(input[i]);
          double i_y = std::get<1>(input[i]);

          double cross = CrossProduct(cur_x, cur_y, next_x, next_y, i_x, i_y);

          if (IsBetterPoint(cross, epsilon, cur_x, cur_y, i_x, i_y, next_x, next_y)) {
            next = i;
            next_x = i_x;
            next_y = i_y;
          }
        }
      }

      void join(const NextPointReduction &other) {
        double cross = CrossProduct(cur_x, cur_y, next_x, next_y, other.next_x, other.next_y);
        if (IsBetterPoint(cross, epsilon, cur_x, cur_y, other.next_x, other.next_y, next_x, next_y)) {
          next = other.next;
          next_x = other.next_x;
          next_y = other.next_y;
        }
      }
    };

    NextPointReduction body(GetInput(), current, current_x, current_y, epsilon);
    tbb::parallel_reduce(tbb::blocked_range<size_t>(0, GetInput().size()), body, tbb::static_partitioner{});

    size_t next = body.next;
    double next_x = body.next_x;
    double next_y = body.next_y;

    current = next;
    current_x = next_x;
    current_y = next_y;

    if (current == leftmost) {
      break;
    }
  }
  return true;
}

bool KutuzovITestConvexHullTBB::PostProcessingImpl() {
  return true;
}

}  // namespace kutuzov_i_convex_hull_jarvis
