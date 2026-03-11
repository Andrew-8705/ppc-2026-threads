#include "redkina_a_integral_simpson_seq/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
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
  size_t dim = a_.size();

  // шаги интегрирования
  std::vector<double> h(dim);
  for (size_t i = 0; i < dim; ++i) {
    h[i] = (b_[i] - a_[i]) / static_cast<double>(n_[i]);
  }

  // произведение шагов
  double h_prod = 1.0;
  for (size_t i = 0; i < dim; ++i) {
    h_prod *= h[i];
  }

  // множители для линеаризации индексов (система счисления со смешанными основаниями)
  std::vector<int> strides(dim);
  strides[dim - 1] = 1;
  for (int i = static_cast<int>(dim) - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * (n_[i + 1] + 1);
  }
  int total_nodes = strides[0] * (n_[0] + 1);

  double sum = 0.0;

  // локальные копии для использования в параллельной области
  const auto &a = a_;
  const auto &n = n_;
  const auto &h_vec = h;
  const auto &func = func_;
  const auto &strides_ref = strides;

#pragma omp parallel default(none) shared(a, n, h_vec, func, strides_ref, total_nodes, dim) reduction(+ : sum)
  {
    std::vector<int> indices(dim);
    std::vector<double> point(dim);

#pragma omp for
    for (int idx = 0; idx < total_nodes; ++idx) {
      // разложение линейного индекса в многомерные индексы
      int remainder = idx;
      for (size_t d = 0; d < dim; ++d) {
        indices[d] = remainder / strides_ref[d];
        remainder = remainder % strides_ref[d];
      }

      // вычисление координат и весов Симпсона
      double w_prod = 1.0;
      for (size_t d = 0; d < dim; ++d) {
        int i = indices[d];
        point[d] = a[d] + i * h_vec[d];
        int w;
        if (i == 0 || i == n[d]) {
          w = 1;
        } else if (i % 2 == 1) {
          w = 4;
        } else {
          w = 2;
        }
        w_prod *= static_cast<double>(w);
      }
      sum += w_prod * func(point);
    }
  }

  // знаменатель 3^dim
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
