#include "frolova_s_radix_sort_double/stl/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ranges>
#include <thread>
#include <utility>
#include <vector>

#include "frolova_s_radix_sort_double/common/include/common.hpp"

namespace frolova_s_radix_sort_double {

constexpr int kRadix = 256;
constexpr int kNumBits = 8;
constexpr int kNumPasses = sizeof(std::uint64_t);

static std::array<int, kRadix> ComputeHistogram(const std::vector<double> &data, int pass, unsigned int num_threads) {
  std::array<std::atomic<int>, kRadix> atomic_histogram{};
  const std::size_t n = data.size();

  std::vector<std::thread> threads;
  const std::size_t chunk_size = (n + num_threads - 1) / num_threads;

  for (unsigned int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    std::size_t start = thread_idx * chunk_size;
    if (start >= n) {
      break;
    }
    std::size_t end = std::min(start + chunk_size, n);

    threads.emplace_back([&, start, end, pass]() {
      for (std::size_t i = start; i < end; ++i) {
        auto bits = std::bit_cast<std::uint64_t>(data[i]);
        int byte = static_cast<int>((bits >> (pass * kNumBits)) & 0xFF);
        atomic_histogram.at(byte).fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  std::array<int, kRadix> histogram{};
  for (int i = 0; i < kRadix; ++i) {
    histogram.at(i) = atomic_histogram.at(i).load();
  }
  return histogram;
}

static std::array<int, kRadix> BuildOffsets(const std::array<int, kRadix> &histogram) {
  std::array<int, kRadix> offsets{};
  int total = 0;
  for (int i = 0; i < kRadix; ++i) {
    offsets.at(i) = total;
    total += histogram.at(i);
  }
  return offsets;
}

static void Distribute(const std::vector<double> &src, std::vector<double> &dst, std::array<int, kRadix> &offsets,
                       int pass) {
  double *dst_ptr = dst.data();
  for (double val : src) {
    auto bits = std::bit_cast<std::uint64_t>(val);
    int byte = static_cast<int>((bits >> (pass * kNumBits)) & 0xFF);
    *(dst_ptr + offsets.at(byte)++) = val;
  }
}

static void FixNegativeOrder(std::vector<double> &data) {
  std::vector<double> negative;
  std::vector<double> positive;
  negative.reserve(data.size());
  positive.reserve(data.size());

  for (double val : data) {
    if (val < 0.0) {
      negative.push_back(val);
    } else {
      positive.push_back(val);
    }
  }

  std::ranges::reverse(negative);

  data.clear();
  data.insert(data.end(), negative.begin(), negative.end());
  data.insert(data.end(), positive.begin(), positive.end());
}

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

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 2;
  }
  if (num_threads > n) {
    num_threads = static_cast<unsigned int>(n);
  }

  std::vector<double> temp(n);

  for (int pass = 0; pass < kNumPasses; ++pass) {
    auto histogram = ComputeHistogram(working, pass, num_threads);
    auto offsets = BuildOffsets(histogram);
    Distribute(working, temp, offsets, pass);
    working.swap(temp);
  }

  FixNegativeOrder(working);

  GetOutput() = std::move(working);
  return true;
}

bool FrolovaSRadixSortDoubleSTL::PostProcessingImpl() {
  return true;
}

}  // namespace frolova_s_radix_sort_double
