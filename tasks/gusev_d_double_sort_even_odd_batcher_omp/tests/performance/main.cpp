#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "gusev_d_double_sort_even_odd_batcher_omp/common/include/common.hpp"
#include "gusev_d_double_sort_even_odd_batcher_omp/omp/include/ops_omp.hpp"

namespace {

using gusev_d_double_sort_even_odd_batcher_omp_task_threads::DoubleSortEvenOddBatcherOMP;
using gusev_d_double_sort_even_odd_batcher_omp_task_threads::InType;
using gusev_d_double_sort_even_odd_batcher_omp_task_threads::OutType;
using gusev_d_double_sort_even_odd_batcher_omp_task_threads::ValueType;

constexpr size_t kPerfInputSize = 1 << 13;

struct PerfRunResult {
  bool ok = false;
  const char *failed_stage = "";
  OutType output;
  std::chrono::duration<double> elapsed{};
};

InType GenerateRandomInput(size_t size, uint64_t seed) {
  std::mt19937_64 generator(seed);
  std::uniform_real_distribution<ValueType> distribution(-1.0e6, 1.0e6);

  InType input(size);
  for (ValueType &value : input) {
    value = distribution(generator);
  }

  return input;
}

InType GenerateDescendingInput(size_t size) {
  InType input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<ValueType>(size - i);
  }

  return input;
}

InType GenerateNearlySortedInput(size_t size) {
  InType input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<ValueType>(i);
  }

  for (size_t i = 1; i < size; i += 64) {
    std::swap(input[i - 1], input[i]);
  }

  return input;
}

InType GenerateDuplicateHeavyInput(size_t size) {
  InType input(size);
  for (size_t i = 0; i < size; ++i) {
    input[i] = static_cast<ValueType>((i % 17) - 8);
  }

  return input;
}

PerfRunResult ExecutePerfCase(const InType &input) {
  DoubleSortEvenOddBatcherOMP task(input);
  if (!task.Validation()) {
    return {.failed_stage = "Validation"};
  }
  if (!task.PreProcessing()) {
    return {.failed_stage = "PreProcessing"};
  }

  const auto started = std::chrono::steady_clock::now();
  if (!task.Run()) {
    return {.failed_stage = "Run"};
  }
  const auto finished = std::chrono::steady_clock::now();

  if (!task.PostProcessing()) {
    return {.failed_stage = "PostProcessing"};
  }

  return {
      .ok = true,
      .output = task.GetOutput(),
      .elapsed = std::chrono::duration<double>(finished - started),
  };
}

void RunPerfCase(const InType &input) {
  auto expected = input;
  std::ranges::sort(expected);

  const auto result = ExecutePerfCase(input);
  ASSERT_TRUE(result.ok) << result.failed_stage;
  EXPECT_EQ(result.output, expected);
  std::cout << "omp_run_time_sec:" << result.elapsed.count() << '\n';
}

TEST(GusevDoubleSortEvenOddBatcherOMPPerf, RunPerfTestOMPDescending) {
  RunPerfCase(GenerateDescendingInput(kPerfInputSize));
}

TEST(GusevDoubleSortEvenOddBatcherOMPPerf, RunPerfTestOMPRandom) {
  RunPerfCase(GenerateRandomInput(kPerfInputSize, 20260320));
}

TEST(GusevDoubleSortEvenOddBatcherOMPPerf, RunPerfTestOMPNearlySorted) {
  RunPerfCase(GenerateNearlySortedInput(kPerfInputSize));
}

TEST(GusevDoubleSortEvenOddBatcherOMPPerf, RunPerfTestOMPDuplicateHeavy) {
  RunPerfCase(GenerateDuplicateHeavyInput(kPerfInputSize));
}

}  // namespace
