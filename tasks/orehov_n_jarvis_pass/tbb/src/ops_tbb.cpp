#include "orehov_n_jarvis_pass/tbb/include/ops_tbb.hpp"

#include <cmath>
#include <cstddef>
#include <set>
#include <vector>

#include "oneapi/tbb.h"
#include "orehov_n_jarvis_pass/common/include/common.hpp"

namespace orehov_n_jarvis_pass {

OrehovNJarvisPassTBB::OrehovNJarvisPassTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<Point>();
}

bool OrehovNJarvisPassTBB::ValidationImpl() {
  return (!GetInput().empty());
}

bool OrehovNJarvisPassTBB::PreProcessingImpl() {
  std::set<Point> tmp(GetInput().begin(), GetInput().end());
  input_.assign(tmp.begin(), tmp.end());
  return true;
}

bool OrehovNJarvisPassTBB::RunImpl() {
  if (input_.size() == 1 || input_.size() == 2) {
    res_ = input_;
    return true;
  }

  Point current = FindFirstElem();
  res_.push_back(current);

  while (true) {
    Point next = FindNext(current);
    if (next == res_[0]) {
      break;
    }

    current = next;
    res_.push_back(next);
  }

  return true;
}

Point OrehovNJarvisPassTBB::FindNext(Point current) const {
  const size_t n = input_.size();

  tbb::combinable<Point> best_point([&]() { return (current == input_[0]) ? input_[1] : input_[0]; });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, n), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      const Point &point = input_[i];
      if (current == point) {
        continue;
      }
      Point &local_best = best_point.local();
      double orient = CheckLeft(current, local_best, point);
      if (orient > 0) {
        local_best = point;
      } else if (orient == 0 && Distance(current, point) > Distance(current, local_best)) {
        local_best = point;
      }
    }
  });

  return best_point.combine([&](Point a, Point b) {
    double orient = CheckLeft(current, a, b);
    if (orient > 0) {
      return b;
    }
    if (orient == 0 && Distance(current, b) > Distance(current, a)) {
      return b;
    }
    return a;
  });
}

double OrehovNJarvisPassTBB::CheckLeft(Point a, Point b, Point c) {
  return ((b.x - a.x) * (c.y - a.y)) - ((b.y - a.y) * (c.x - a.x));
}

Point OrehovNJarvisPassTBB::FindFirstElem() const {
  Point current = input_[0];
  for (auto f : input_) {
    if (f.x < current.x || (f.y < current.y && f.x == current.x)) {
      current = f;
    }
  }
  return current;
}

double OrehovNJarvisPassTBB::Distance(Point a, Point b) {
  return std::sqrt(std::pow(a.y - b.y, 2) + std::pow(a.x - b.x, 2));
}

bool OrehovNJarvisPassTBB::PostProcessingImpl() {
  GetOutput() = res_;
  return true;
}

}  // namespace orehov_n_jarvis_pass
