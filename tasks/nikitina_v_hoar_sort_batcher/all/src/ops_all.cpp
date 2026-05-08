#include "nikitina_v_hoar_sort_batcher/all/include/ops_all.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <vector>

#include "nikitina_v_hoar_sort_batcher/common/include/common.hpp"

namespace {

void QuickSortHoare(std::vector<int> &arr, int low, int high) {
  if (low >= high) {
    return;
  }
  int pivot = arr[low + (high - low) / 2];
  int i = low - 1;
  int j = high + 1;
  while (true) {
    while (arr[++i] < pivot) {
    }
    while (arr[--j] > pivot) {
    }
    if (i >= j) {
      break;
    }
    std::swap(arr[i], arr[j]);
  }
  QuickSortHoare(arr, low, j);
  QuickSortHoare(arr, j + 1, high);
}

void OddEvenMerge(std::vector<int> &arr, int l, int r, int step) {
  int m = step * 2;
  if (m < r - l) {
    OddEvenMerge(arr, l, r, m);
    OddEvenMerge(arr, l + step, r, m);
    for (int i = l + step; i + step < r; i += m) {
      if (arr[i] > arr[i + step]) {
        std::swap(arr[i], arr[i + step]);
      }
    }
  } else {
    if (l + step < r && arr[l] > arr[l + step]) {
      std::swap(arr[l], arr[l + step]);
    }
  }
}

void ParallelSort(std::vector<int> &arr, int num_threads) {
  if (arr.empty() || num_threads <= 0) {
    return;
  }
  int n = arr.size();
  int pad_size = 1;
  while (pad_size < n) {
    pad_size *= 2;
  }
  arr.resize(pad_size, std::numeric_limits<int>::max());
  int active_threads = 1;
  while (active_threads * 2 <= num_threads) {
    active_threads *= 2;
  }
  int chunk = pad_size / active_threads;
  if (chunk == 0) {
    chunk = 1;
  }
  std::vector<std::future<void>> futures;
  for (int i = 0; i < active_threads; ++i) {
    int start = i * chunk;
    int end = start + chunk;
    if (start >= pad_size) {
      break;
    }
    futures.push_back(std::async(std::launch::async, [&arr, start, end]() { QuickSortHoare(arr, start, end - 1); }));
  }
  for (auto &f : futures) {
    f.get();
  }
  for (int step = chunk; step < pad_size; step *= 2) {
    std::vector<std::future<void>> merge_futures;
    for (int t = 0; t < active_threads; ++t) {
      merge_futures.push_back(std::async(std::launch::async, [&arr, t, active_threads, pad_size, step]() {
        for (int i = t * 2 * step; i < pad_size; i += active_threads * 2 * step) {
          OddEvenMerge(arr, i, i + 2 * step, 1);
        }
      }));
    }
    for (auto &f : merge_futures) {
      f.get();
    }
  }
  arr.resize(n);
}

}  // namespace

namespace nikitina_v_hoar_sort_batcher {

HoareSortBatcherALL::HoareSortBatcherALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool HoareSortBatcherALL::ValidationImpl() {
  return true;
}

bool HoareSortBatcherALL::PreProcessingImpl() {
  data_ = GetInput();
  return true;
}

bool HoareSortBatcherALL::RunImpl() {
  int num_threads = ppc::util::GetNumThreads();
  if (num_threads <= 0) {
    num_threads = 1;
  }
  ParallelSort(data_, num_threads);
  return true;
}

bool HoareSortBatcherALL::PostProcessingImpl() {
  GetOutput() = data_;
  return true;
}

}  // namespace nikitina_v_hoar_sort_batcher
