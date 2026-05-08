#include "nikitina_v_hoar_sort_batcher/all/include/ops_all.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <vector>

#include "nikitina_v_hoar_sort_batcher/common/include/common.hpp"

namespace nikitina_v_hoar_sort_batcher {

namespace {

void QuickSortHoare(std::vector<int> &arr, int low, int high) {
  if (low >= high) {
    return;
  }
  std::vector<std::pair<int, int>> stack;
  stack.emplace_back(low, high);

  while (!stack.empty()) {
    auto [left_bound, right_bound] = stack.back();
    stack.pop_back();

    if (left_bound >= right_bound) {
      continue;
    }

    int pivot = arr[left_bound + ((right_bound - left_bound) / 2)];
    int left_idx = left_bound - 1;
    int right_idx = right_bound + 1;

    while (true) {
      left_idx++;
      while (arr[left_idx] < pivot) {
        left_idx++;
      }

      right_idx--;
      while (arr[right_idx] > pivot) {
        right_idx--;
      }

      if (left_idx >= right_idx) {
        break;
      }
      std::swap(arr[left_idx], arr[right_idx]);
    }

    stack.emplace_back(left_bound, right_idx);
    stack.emplace_back(right_idx + 1, right_bound);
  }
}

void CompareSplit(std::vector<int> &arr, int start_first, int len_first, int start_second, int len_second) {
  if (len_first == 0 || len_second == 0) {
    return;
  }

  std::vector<int> left_block(arr.begin() + start_first, arr.begin() + start_first + len_first);
  std::vector<int> right_block(arr.begin() + start_second, arr.begin() + start_second + len_second);

  int ptr1 = 0;
  int ptr2 = 0;
  int write1 = start_first;
  int write2 = start_second;

  for (int iter = 0; iter < len_first + len_second; ++iter) {
    int val = 0;
    if (ptr1 < len_first && (ptr2 == len_second || left_block[ptr1] <= right_block[ptr2])) {
      val = left_block[ptr1++];
    } else {
      val = right_block[ptr2++];
    }

    if (iter < len_first) {
      arr[write1++] = val;
    } else {
      arr[write2++] = val;
    }
  }
}

void BuildPairs(std::vector<std::pair<int, int>> &pairs, int num_threads, int step_p, int step_k) {
  for (int idx_j = step_k % step_p; idx_j + step_k < num_threads; idx_j += (step_k * 2)) {
    for (int idx_i = 0; idx_i < std::min(step_k, num_threads - idx_j - step_k); idx_i++) {
      if ((idx_j + idx_i) / (step_p * 2) == (idx_j + idx_i + step_k) / (step_p * 2)) {
        pairs.emplace_back(idx_j + idx_i, idx_j + idx_i + step_k);
      }
    }
  }
}

void ExecuteMergeStep(std::vector<int> &output, const std::vector<int> &offsets,
                      const std::vector<std::pair<int, int>> &pairs, int actual_threads) {
  int num_pairs = static_cast<int>(pairs.size());
  int chunk_size = (num_pairs + actual_threads - 1) / actual_threads;
  std::vector<std::thread> threads;
  threads.reserve(actual_threads);

  for (int thread_idx = 0; thread_idx < actual_threads; ++thread_idx) {
    int start = thread_idx * chunk_size;
    int end = std::min(start + chunk_size, num_pairs);

    if (start < end) {
      threads.emplace_back([start, end, &output, &offsets, &pairs]() {
        for (int idx = start; idx < end; ++idx) {
          int block_a = pairs[idx].first;
          int block_b = pairs[idx].second;
          CompareSplit(output, offsets[block_a], offsets[block_a + 1] - offsets[block_a], offsets[block_b],
                       offsets[block_b + 1] - offsets[block_b]);
        }
      });
    }
  }

  for (auto &th : threads) {
    th.join();
  }
}

void BatcherMergePhase(std::vector<int> &output, const std::vector<int> &offsets, int num_threads, int hw_threads) {
  for (int step_p = 1; step_p < num_threads; step_p *= 2) {
    for (int step_k = step_p; step_k > 0; step_k /= 2) {
      std::vector<std::pair<int, int>> pairs;
      BuildPairs(pairs, num_threads, step_p, step_k);

      int num_pairs = static_cast<int>(pairs.size());
      if (num_pairs == 0) {
        continue;
      }

      int actual_threads = std::min(hw_threads, num_pairs);
      ExecuteMergeStep(output, offsets, pairs, actual_threads);
    }
  }
}

}  // namespace

HoareSortBatcherALL::HoareSortBatcherALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool HoareSortBatcherALL::ValidationImpl() {
  return true;
}

bool HoareSortBatcherALL::PreProcessingImpl() {
  GetOutput() = GetInput();
  return true;
}

bool HoareSortBatcherALL::RunImpl() {
  auto &output = GetOutput();

  int orig_n = static_cast<int>(output.size());
  if (orig_n <= 1) {
    return true;
  }

  int hw_threads = ppc::util::GetNumThreads();
  if (hw_threads <= 0) {
    hw_threads = 4;
  }

  int active_blocks = 1;
  while (active_blocks * 2 <= hw_threads && active_blocks * 2 <= orig_n) {
    active_blocks *= 2;
  }

  if (active_blocks == 1) {
    QuickSortHoare(output, 0, orig_n - 1);
    return true;
  }

  int pad = (active_blocks - (orig_n % active_blocks)) % active_blocks;
  for (int iter = 0; iter < pad; ++iter) {
    output.push_back(std::numeric_limits<int>::max());
  }

  int total_padded_n = orig_n + pad;
  std::vector<int> offsets(active_blocks + 1, 0);
  int chunk = total_padded_n / active_blocks;
  for (int iter = 0; iter <= active_blocks; ++iter) {
    offsets[iter] = iter * chunk;
  }

  std::vector<std::thread> sort_threads;
  sort_threads.reserve(active_blocks);

  for (int iter = 0; iter < active_blocks; ++iter) {
    sort_threads.emplace_back(
        [&output, &offsets, iter]() { QuickSortHoare(output, offsets[iter], offsets[iter + 1] - 1); });
  }

  for (auto &th : sort_threads) {
    th.join();
  }

  BatcherMergePhase(output, offsets, active_blocks, hw_threads);

  output.resize(orig_n);

  return true;
}

bool HoareSortBatcherALL::PostProcessingImpl() {
  return true;
}

}  // namespace nikitina_v_hoar_sort_batcher
