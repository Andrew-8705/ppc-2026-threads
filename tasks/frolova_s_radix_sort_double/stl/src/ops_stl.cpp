#include "frolova_s_radix_sort_double/stl/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <thread>
#include <vector>

#include "frolova_s_radix_sort_double/common/include/common.hpp"

namespace frolova_s_radix_sort_double {

FrolovaSRadixSortDoubleSTL::FrolovaSRadixSortDoubleSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool FrolovaSRadixSortDoubleSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool FrolovaSRadixSortDoubleSTL::PreProcessingImpl() {
  return true;
}

bool FrolovaSRadixSortDoubleSTL::RunImpl() {
  const std::vector<double> &input = GetInput();
  if (input.empty()) {
    return false;
  }

  std::vector<double> working = input;
  const std::size_t n = working.size();

  constexpr int kRadix = 256;
  constexpr int kNumBits = 8;
  constexpr int kNumPasses = sizeof(std::uint64_t);

  std::vector<double> temp(n);

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2;
  }
  if (num_threads > n) {
    num_threads = static_cast<unsigned int>(n);
  }

  for (int pass = 0; pass < kNumPasses; ++pass) {
    std::array<std::atomic<int>, kRadix> count{};
    {
      std::vector<std::thread> threads;
      std::size_t chunk_size = (n + num_threads - 1) / num_threads;
      for (unsigned int t = 0; t < num_threads; ++t) {
        std::size_t start = t * chunk_size;
        if (start >= n) {
          break;
        }
        std::size_t end = std::min(start + chunk_size, n);
        threads.emplace_back([&, start, end]() {
          for (std::size_t i = start; i < end; ++i) {
            auto bits = std::bit_cast<std::uint64_t>(working[i]);
            int byte = static_cast<int>((bits >> (pass * kNumBits)) & 0xFF);
            count[byte].fetch_add(1, std::memory_order_relaxed);
          }
        });
      }
      for (auto &t : threads) {
        t.join();
      }
    }
    std::array<int, kRadix> offset{};
    int total = 0;
    for (int i = 0; i < kRadix; ++i) {
      offset[i] = total;
      total += count[i].load();
    }

    for (double val : working) {
      auto bits = std::bit_cast<std::uint64_t>(val);
      int byte = static_cast<int>((bits >> (pass * kNumBits)) & 0xFF);
      temp[offset[byte]++] = val;
    }

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

bool FrolovaSRadixSortDoubleSTL::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_radix_sort_double
