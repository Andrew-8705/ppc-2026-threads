#include "lukin_i_ench_contr_lin_hist/stl/include/ops_stl.hpp"

#include <algorithm>
#include <thread>
#include <vector>

#include "lukin_i_ench_contr_lin_hist/common/include/common.hpp"

namespace lukin_i_ench_contr_lin_hist {

LukinITestTaskSTL::LukinITestTaskSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType(GetInput().size());
}

bool LukinITestTaskSTL::ValidationImpl() {
  return !(GetInput().empty());
}

bool LukinITestTaskSTL::PreProcessingImpl() {
  return true;
}

bool LukinITestTaskSTL::RunImpl() {
  const auto &input = GetInput();
  auto &output = GetOutput();

  const int size = static_cast<int>(input.size());

  const int thread_count = std::thread::hardware_concurrency();
  const int chunk_size = input.size() / thread_count;

  std::vector<unsigned char> loc_mins(thread_count, 255);
  std::vector<unsigned char> loc_maxs(thread_count, 0);

  std::vector<std::thread> thread_pool;
  thread_pool.reserve(thread_count);

  auto reduction = [&](const int idx, const int start, const int end) {
    int loc_min = loc_mins[idx];
    int loc_max = loc_maxs[idx];

    for (int i = start; i < end; i++) {
      loc_min = (input[i] < loc_min) ? input[i] : loc_min;
      loc_max = (input[i] > loc_max) ? input[i] : loc_max;
    }

    loc_mins[idx] = loc_min;
    loc_maxs[idx] = loc_max;
  };

  for (int i = 0; i < thread_count; i++) {
    const int start = chunk_size * i;
    const int end = (i == (thread_count - 1)) ? size : start + chunk_size;
    thread_pool.emplace_back(reduction, i, start, end);
  }
  for (auto &thread : thread_pool) {
    thread.join();
  }
  thread_pool.clear();

  const int min = *std::min_element(loc_mins.begin(), loc_mins.end());
  const int max = *std::max_element(loc_maxs.begin(), loc_maxs.end());

  if (max == min) {
    GetOutput() = GetInput();
    return true;
  }

  const float scale = 255.0F / static_cast<float>(max - min);

  auto process = [&](const int start, const int end) {
    for (int i = start; i < end; i++) {
      output[i] = static_cast<unsigned char>(static_cast<float>(input[i] - min) * scale);
    }
  };

  for (int i = 0; i < thread_count; i++) {
    const int start = chunk_size * i;
    const int end = (i == (thread_count - 1)) ? size : start + chunk_size;
    thread_pool.emplace_back(process, start, end);
  }
  for (auto &thread : thread_pool) {
    thread.join();
  }

  return true;
}

bool LukinITestTaskSTL::PostProcessingImpl() {
  return true;
}

}  // namespace lukin_i_ench_contr_lin_hist
