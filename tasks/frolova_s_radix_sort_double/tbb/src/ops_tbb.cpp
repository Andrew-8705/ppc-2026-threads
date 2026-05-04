#include "frolova_s_radix_sort_double/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include "frolova_s_radix_sort_double/common/include/common.hpp"

namespace frolova_s_radix_sort_double {

FrolovaSRadixSortDoubleTBB::FrolovaSRadixSortDoubleTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool FrolovaSRadixSortDoubleTBB::ValidationImpl() {
  return !GetInput().empty();
}

bool FrolovaSRadixSortDoubleTBB::PreProcessingImpl() {
  return true;
}

bool FrolovaSRadixSortDoubleTBB::RunImpl() {
  const std::vector<double> &input = GetInput();
  if (input.empty()) {
    return false;
  }

  std::vector<double> working = input;
  const std::size_t n = working.size();

  constexpr int kRadix = 256;
  constexpr int kBits = 8;
  constexpr int kPasses = sizeof(std::uint64_t);

  std::vector<double> temp(n);

  const std::size_t num_threads = tbb::this_task_arena::max_concurrency();
  const std::size_t grain_size = std::max(std::size_t(256), n / (num_threads * 8));
  const std::size_t num_blocks = (n + grain_size - 1) / grain_size;
  std::vector<std::size_t> block_starts(num_blocks + 1);
  for (std::size_t i = 0; i < num_blocks; ++i) {
    block_starts[i] = i * grain_size;
  }
  block_starts[num_blocks] = n;

  for (int pass = 0; pass < kPasses; ++pass) {
    std::vector<std::array<std::size_t, kRadix>> block_counts(num_blocks, std::array<std::size_t, kRadix>{});

    tbb::parallel_for(std::size_t(0), num_blocks, [&](std::size_t block) {
      std::size_t start = block_starts[block];
      std::size_t end = block_starts[block + 1];
      for (std::size_t i = start; i < end; ++i) {
        auto bits = std::bit_cast<std::uint64_t>(working[i]);
        int byte = static_cast<int>((bits >> (pass * kBits)) & 0xFF);
        ++block_counts[block][byte];
      }
    });

    std::vector<std::array<std::size_t, kRadix>> block_offsets(num_blocks);
    std::array<std::size_t, kRadix> current_offset{};
    for (std::size_t block = 0; block < num_blocks; ++block) {
      for (int byte = 0; byte < kRadix; ++byte) {
        block_offsets[block][byte] = current_offset[byte];
        current_offset[byte] += block_counts[block][byte];
      }
    }

    tbb::parallel_for(std::size_t(0), num_blocks, [&](std::size_t block) {
      std::array<std::size_t, kRadix> local_pos = block_offsets[block];
      std::size_t start = block_starts[block];
      std::size_t end = block_starts[block + 1];
      for (std::size_t i = start; i < end; ++i) {
        auto bits = std::bit_cast<std::uint64_t>(working[i]);
        int byte = static_cast<int>((bits >> (pass * kBits)) & 0xFF);
        temp[local_pos[byte]++] = working[i];
      }
    });

    working.swap(temp);
  }

  std::vector<double> negative;
  std::vector<double> positive;
  negative.reserve(n);
  positive.reserve(n);

  for (double val : working) {
    if (val < 0.0) {
      negative.push_back(val);
    } else {
      positive.push_back(val);
    }
  }
  std::reverse(negative.begin(), negative.end());

  working.clear();
  working.insert(working.end(), negative.begin(), negative.end());
  working.insert(working.end(), positive.begin(), positive.end());

  GetOutput() = std::move(working);
  return true;
}

bool FrolovaSRadixSortDoubleTBB::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_radix_sort_double
