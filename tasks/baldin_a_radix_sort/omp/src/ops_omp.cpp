#include "baldin_a_radix_sort/omp/include/ops_omp.hpp"

#include <omp.h>

#include <atomic>
#include <numeric>
#include <vector>

#include "baldin_a_radix_sort/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_radix_sort {

BaldinARadixSortOMP::BaldinARadixSortOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BaldinARadixSortOMP::ValidationImpl() {
  return true;
}

bool BaldinARadixSortOMP::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

namespace {

void CountingSortStep(std::vector<int>::iterator in_begin, std::vector<int>::iterator in_end,
                      std::vector<int>::iterator out_begin, size_t byte_index) {
  size_t shift = byte_index * 8;
  size_t count[256] = {0};

  for (auto it = in_begin; it != in_end; it++) {
    auto raw_val = static_cast<unsigned int>(*it);
    unsigned int byte_val = (raw_val >> shift) & 0xFF;

    if (byte_index == sizeof(int) - 1) {
      byte_val ^= 128;
    }
    count[byte_val]++;
  }

  size_t prefix[256];
  prefix[0] = 0;
  for (int i = 1; i < 256; i++) {
    prefix[i] = prefix[i - 1] + count[i - 1];
  }

  for (auto it = in_begin; it != in_end; it++) {
    auto raw_val = static_cast<unsigned int>(*it);
    unsigned int byte_val = (raw_val >> shift) & 0xFF;

    if (byte_index == sizeof(int) - 1) {
      byte_val ^= 128;
    }

    *(out_begin + prefix[byte_val]) = *it;
    prefix[byte_val]++;
  }
}

void RadixSortLocal(std::vector<int>::iterator begin, std::vector<int>::iterator end) {
  size_t n = std::distance(begin, end);
  if (n < 2) {
    return;
  }

  std::vector<int> temp(n);

  for (size_t i = 0; i < sizeof(int); i++) {
    size_t shift = i;

    if (i % 2 == 0) {
      CountingSortStep(begin, end, temp.begin(), shift);
    } else {
      CountingSortStep(temp.begin(), temp.end(), begin, shift);
    }
  }
}

}  // namespace

bool BaldinARadixSortOMP::RunImpl() {
  int n = static_cast<int>(GetOutput().size());
  if (n == 0) {
    return true;
  }

  int num_threads = ppc::util::GetNumThreads();
  // int num_threads = omp_get_max_threads();
  // std::cout << "NUM THREADS: " << num_threads << '\n';

  if (n < num_threads * 100) {
    num_threads = 1;
  }

  if (num_threads == 1) {
    RadixSortLocal(GetOutput().begin(), GetOutput().end());
    return true;
  }

  std::vector<int> offsets(num_threads + 1);
  int chunk = n / num_threads;
  int rem = n % num_threads;
  int curr = 0;
  for (int i = 0; i < num_threads; ++i) {
    offsets[i] = curr;
    curr += chunk + (i < rem ? 1 : 0);
  }
  offsets[num_threads] = n;

#pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    auto begin = GetOutput().begin() + offsets[tid];
    auto end = GetOutput().begin() + offsets[tid + 1];

    RadixSortLocal(begin, end);
  }

  for (int step = 1; step < num_threads; step *= 2) {
#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; i += 2 * step) {
      if (i + step < num_threads) {
        auto begin = GetOutput().begin() + offsets[i];
        auto middle = GetOutput().begin() + offsets[i + step];

        int end_idx = std::min(i + 2 * step, num_threads);
        auto end = GetOutput().begin() + offsets[end_idx];

        std::inplace_merge(begin, middle, end);
      }
    }
  }

  return true;
}

bool BaldinARadixSortOMP::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_radix_sort
