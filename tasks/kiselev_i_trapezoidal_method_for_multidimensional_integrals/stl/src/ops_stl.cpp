#include "kiselev_i_trapezoidal_method_for_multidimensional_integrals/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <future>
#include <vector>

#include "kiselev_i_trapezoidal_method_for_multidimensional_integrals/common/include/common.hpp"
#include "util/include/util.hpp"

namespace kiselev_i_trapezoidal_method_for_multidimensional_integrals {

KiselevITestTaskSTL::KiselevITestTaskSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0.0;
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

    case 4:
      return x + y;

    default:
      return x + y;
  }
}

double KiselevITestTaskSTL::ComputeIntegral(const std::vector<int> &steps) {
  const auto &input_data = GetInput();

  const double hx =
      static_cast<double>(input_data.right_bounds[0] - input_data.left_bounds[0]) / static_cast<double>(steps[0]);

  const double hy =
      static_cast<double>(input_data.right_bounds[1] - input_data.left_bounds[1]) / static_cast<double>(steps[1]);

  double result = 0.0;

  for (int x_index = 0; x_index <= steps[0]; x_index++) {
    const double x = input_data.left_bounds[0] + (static_cast<double>(x_index) * hx);

    const double weight_x = (x_index == 0 || x_index == steps[0]) ? 0.5 : 1.0;

    for (int y_index = 0; y_index <= steps[1]; y_index++) {
      const double y = input_data.left_bounds[1] + (static_cast<double>(y_index) * hy);

      const double weight_y = (y_index == 0 || y_index == steps[1]) ? 0.5 : 1.0;

      const double value = FunctionTypeChoose(input_data.type_function, x, y);

      result += weight_x * weight_y * value;
    }
  }

  return result * hx * hy;
}

bool KiselevITestTaskSTL::RunImpl() {
  const auto &input_data = GetInput();

  if (input_data.left_bounds.size() != 2 || input_data.right_bounds.size() != 2 || input_data.step_n_size.size() != 2) {
    GetOutput() = 0.0;
    return true;
  }

  std::vector<int> steps = input_data.step_n_size;

  for (const auto &step : steps) {
    if (step <= 0) {
      GetOutput() = 0.0;
      return true;
    }
  }

  const double epsilon = input_data.epsilon;

  if (epsilon <= 0.0) {
    GetOutput() = ComputeIntegral(steps);
    return true;
  }

  double previous_result = ComputeIntegral(steps);
  double current_result = previous_result;

  const int max_iterations = 10;

  for (int iteration = 0; iteration < max_iterations; iteration++) {
    for (auto &step : steps) {
      step *= 2;
    }

    current_result = ComputeIntegral(steps);

    if (std::abs(current_result - previous_result) < epsilon) {
      break;
    }

    previous_result = current_result;
  }

  GetOutput() = current_result;

  return true;
}

bool KiselevITestTaskSTL::PostProcessingImpl() {
  return true;
}

}  // namespace kiselev_i_trapezoidal_method_for_multidimensional_integrals
