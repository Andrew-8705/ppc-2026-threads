#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cstddef>
#include <functional>
#include <vector>

#include "redkina_a_integral_simpson_seq/common/include/common.hpp"

namespace redkina_a_integral_simpson_seq {

RedkinaAIntegralSimpsonOMP::RedkinaAIntegralSimpsonOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool RedkinaAIntegralSimpsonOMP::ValidationImpl() {
  const auto &in = GetInput();
  size_t dim = in.a.size();

  if (dim == 0 || in.b.size() != dim || in.n.size() != dim) {
    return false;
  }

  for (size_t i = 0; i < dim; ++i) {
    if (in.a[i] >= in.b[i]) {
      return false;
    }
    if (in.n[i] <= 0 || in.n[i] % 2 != 0) {
      return false;
    }
  }

  return static_cast<bool>(in.func);
}

bool RedkinaAIntegralSimpsonOMP::PreProcessingImpl() {
  const auto &in = GetInput();

  func_ = in.func;
  a_ = in.a;
  b_ = in.b;
  n_ = in.n;

  result_ = 0.0;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::RunImpl() {
  const size_t dim = a_.size();

  std::vector<double> h(dim);
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_[i] - a_[i]) / static_cast<double>(n_[i]);
  }

  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h_prod *= h[i];
  }

  size_t total_points = 1;
  for (size_t i = 0; i < dim; ++i) {
    total_points *= static_cast<size_t>(n_[i] + 1);
  }

  double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
  for (long long linear = 0; linear < static_cast<long long>(total_points); ++linear) {
    std::vector<int> indices(dim);
    long long tmp = linear;

    for (int d = static_cast<int>(dim) - 1; d >= 0; --d) {
      int size_d = n_[d] + 1;
      indices[d] = static_cast<int>(tmp % size_d);
      tmp /= size_d;
    }

    std::vector<double> point(dim);

    double w_prod = 1.0;

    for (size_t d = 0; d < dim; ++d) {
      int idx = indices[d];

      point[d] = a_[d] + (static_cast<double>(idx) * h[d]);

      int w = 0;

      if (idx == 0 || idx == n_[d]) {
        w = 1;
      } else if (idx % 2 == 1) {
        w = 4;
      } else {
        w = 2;
      }

      w_prod *= static_cast<double>(w);
    }

    sum += w_prod * func_(point);
  }

  double denominator = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    denominator *= 3.0;
  }

  result_ = (h_prod / denominator) * sum;

  return true;
}

bool RedkinaAIntegralSimpsonOMP::PostProcessingImpl() {
  GetOutput() = result_;
  return true;
}

}  // namespace redkina_a_integral_simpson_seq
