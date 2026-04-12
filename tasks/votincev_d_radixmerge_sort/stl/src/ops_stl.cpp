#include "votincev_d_radixmerge_sort/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>
#include <vector>

namespace votincev_d_radixmerge_sort {

VotincevDRadixMergeSortSTL::VotincevDRadixMergeSortSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VotincevDRadixMergeSortSTL::ValidationImpl() {
  return !GetInput().empty();
}

bool VotincevDRadixMergeSortSTL::PreProcessingImpl() {
  return true;
}

// поразрядная сортировка для локальных блоков (LSD)
void VotincevDRadixMergeSortSTL::LocalRadixSort(uint32_t *begin, uint32_t *end) {
  int32_t n = static_cast<int32_t>(end - begin);
  if (n <= 1) {
    return;
  }

  uint32_t max_val = *std::max_element(begin, end);

  std::vector<uint32_t> buffer(n);
  uint32_t *src = begin;
  uint32_t *dst = buffer.data();

  for (int64_t exp = 1; static_cast<int64_t>(max_val) / exp > 0; exp *= 10) {
    int32_t count[10] = {0};

    for (int32_t i = 0; i < n; ++i) {
      count[(src[i] / exp) % 10]++;
    }
    for (int32_t i = 1; i < 10; ++i) {
      count[i] += count[i - 1];
    }
    for (int32_t i = n - 1; i >= 0; --i) {
      uint32_t digit = (src[i] / exp) % 10;
      dst[--count[digit]] = src[i];
    }
    std::swap(src, dst);
  }

  if (src != begin) {
    std::copy(src, src + n, begin);
  }
}

// слияние двух отсортированных участков
void VotincevDRadixMergeSortSTL::Merge(uint32_t *data, int32_t left, int32_t mid, int32_t right, uint32_t *temp) {
  int32_t i = left, j = mid, k = left;
  while (i < mid && j < right) {
    temp[k++] = (data[i] <= data[j]) ? data[i++] : data[j++];
  }
  while (i < mid) {
    temp[k++] = data[i++];
  }
  while (j < right) {
    temp[k++] = data[j++];
  }

  std::copy(temp + left, temp + right, data + left);
}

// параллельная сортировка слиянием через std::async
void VotincevDRadixMergeSortSTL::ParallelRadixMergeSort(uint32_t *data, int32_t left, int32_t right, uint32_t *temp) {
  const int32_t GRAIN_SIZE = 4096;  // порог для перехода на последовательную сортировку

  if (right - left <= GRAIN_SIZE) {
    LocalRadixSort(data + left, data + right);
    return;
  }

  int32_t mid = left + (right - left) / 2;

  // запускаем левую часть в отдельном потоке (аналог tbb::parallel_invoke)
  auto future = std::async(std::launch::async, [&] { ParallelRadixMergeSort(data, left, mid, temp); });

  // правую часть выполняем в текущем потоке
  ParallelRadixMergeSort(data, mid, right, temp);

  // ждем завершения левой части
  future.get();

  // сливаем результаты
  Merge(data, left, mid, right, temp);
}

bool VotincevDRadixMergeSortSTL::RunImpl() {
  const auto &input = GetInput();
  int32_t n = static_cast<int32_t>(input.size());

  // поиск минимума
  int32_t min_val = *std::min_element(input.begin(), input.end());

  // uint32_t чтобы избежать проблем с отрицательными числами
  std::vector<uint32_t> working_array(n);
  for (int32_t i = 0; i < n; ++i) {
    working_array[i] = static_cast<uint32_t>(input[i]) - static_cast<uint32_t>(min_val);
  }

  // параллельная сортировка
  std::vector<uint32_t> temp_buffer(n);
  ParallelRadixMergeSort(working_array.data(), 0, n, temp_buffer.data());

  // восстановление исходных значений
  std::vector<int32_t> result(n);
  for (int32_t i = 0; i < n; ++i) {
    result[i] = static_cast<int32_t>(working_array[i] + static_cast<uint32_t>(min_val));
  }

  GetOutput() = std::move(result);
  return true;
}

bool VotincevDRadixMergeSortSTL::PostProcessingImpl() {
  return std::ranges::is_sorted(GetOutput());
}

}  // namespace votincev_d_radixmerge_sort
