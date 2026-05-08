#include "kiselev_i_trapezoidal_method_for_multidimensional_integrals/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <vector>

#include "kiselev_i_trapezoidal_method_for_multidimensional_integrals/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_trapezoidal_method_for_multidimensional_integrals {

KiselevITestTaskSTL::KiselevITestTaskSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool KiselevITestTaskSTL::ValidationImpl() {
  return true;
}

bool KiselevITestTaskSTL::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

double KiselevITestTaskSTL::FunctionTypeChoose(int type_x, double x, double y) {
  switch (type_x) {
    case 0:
      return (x * x) + (y * y);

    case 1:
      return std::sin(x) * std::cos(y);

    case 2:
      return std::sin(x) + std::cos(y);

    case 3:
      return std::exp(x + y);

    default:
      return x + y;
  }
}

double KiselevITestTaskSTL::ComputeIntegral(const std::vector<int> &steps) {
  const auto &in = GetInput();

  const double hx = (in.right_bounds[0] - in.left_bounds[0]) / static_cast<double>(steps[0]);

  const double hy = (in.right_bounds[1] - in.left_bounds[1]) / static_cast<double>(steps[1]);

  int num_threads = ppc::util::GetNumThreads();

  if (num_threads <= 0) {
    num_threads = 1;
  }

  const int total_iters = steps[0] + 1;

  // Не создаем больше потоков, чем работы
  num_threads = std::min(num_threads, total_iters);

  const int chunk_size = total_iters / num_threads;
  const int remainder = total_iters % num_threads;

  std::vector<std::future<double>> futures;
  futures.reserve(num_threads);

  int start = 0;

  for (int t = 0; t < num_threads; t++) {
    int end = start + chunk_size;

    // равномернее распределяем remainder
    if (t < remainder) {
      end++;
    }

    futures.emplace_back(std::async(std::launch::async, [this, &in, &steps, hx, hy, start, end]() {
      double local_result = 0.0;

      for (int i = start; i < end; i++) {
        const double x = in.left_bounds[0] + (i * hx);
        const double wx = (i == 0 || i == steps[0]) ? 0.5 : 1.0;

        for (int j = 0; j <= steps[1]; j++) {
          const double y = in.left_bounds[1] + (j * hy);

          const double wy = (j == 0 || j == steps[1]) ? 0.5 : 1.0;

          local_result += wx * wy * FunctionTypeChoose(in.type_function, x, y);
        }
      }

      return local_result;
    }));

    start = end;
  }

  double result = 0.0;

  for (auto &future : futures) {
    result += future.get();
  }

  return result * hx * hy;
}

bool KiselevITestTaskSTL::RunImpl() {
  std::vector<int> steps = GetInput().step_n_size;
  double epsilon = GetInput().epsilon;

  const auto &in = GetInput();

  if (in.left_bounds.size() != 2 || in.right_bounds.size() != 2 || in.step_n_size.size() != 2) {
    GetOutput() = 0.0;
    return true;
  }

  if (epsilon <= 0.0) {
    GetOutput() = ComputeIntegral(steps);
    return true;
  }

  double prev = ComputeIntegral(steps);
  double current = prev;

  int iter = 0;
  const int max_iter = 1;

  while (iter < max_iter) {
    for (auto &s : steps) {
      s *= 2;
    }

    current = ComputeIntegral(steps);

    if (std::abs(current - prev) < epsilon) {
      break;
    }

    prev = current;
    iter++;
  }

  GetOutput() = current;

  return true;
}

bool KiselevITestTaskSTL::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_trapezoidal_method_for_multidimensional_integrals
