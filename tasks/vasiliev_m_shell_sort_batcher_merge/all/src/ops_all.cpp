#include "vasiliev_m_shell_sort_batcher_merge/all/include/ops_all.hpp"

#include <omp.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#include "util/include/util.hpp"
#include "vasiliev_m_shell_sort_batcher_merge/common/include/common.hpp"

namespace vasiliev_m_shell_sort_batcher_merge {

VasilievMShellSortBatcherMergeALL::VasilievMShellSortBatcherMergeALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMShellSortBatcherMergeALL::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMShellSortBatcherMergeALL::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VasilievMShellSortBatcherMergeALL::RunImpl() {
  auto &vec = GetInput();
  const size_t n = vec.size();

  if (vec.empty()) {
    return false;
  }

  int threads = std::max(1, ppc::util::GetNumThreads());
  std::vector<size_t> bounds = ChunkBoundaries(n, threads);
  const size_t chunk_count = bounds.size() - 1;

  ShellSortOMP(vec, bounds, threads);

  std::vector<ValType> buffer(n);
  size_t size = 1;
  for (; size * 2 < chunk_count; size *= 2) {
    CycleMergeTBB(vec, buffer, bounds, size);
    vec.swap(buffer);
  }

  for (; size < chunk_count; size *= 2) {
    CycleMergeSTL(vec, buffer, bounds, size, threads);
    vec.swap(buffer);
  }

  GetOutput() = vec;
  return true;
}

bool VasilievMShellSortBatcherMergeALL::PostProcessingImpl() {
  return true;
}

std::vector<size_t> VasilievMShellSortBatcherMergeALL::ChunkBoundaries(size_t vec_size, size_t threads) {
  size_t chunks = std::max<size_t>(1, std::min(threads, vec_size));
  std::vector<size_t> bounds;
  bounds.reserve(chunks + 1);

  size_t chunk_size = vec_size / chunks;
  size_t remainder = vec_size % chunks;
  bounds.push_back(0);

  for (size_t i = 0; i < chunks; i++) {
    bounds.push_back(bounds.back() + chunk_size + (i < remainder ? 1 : 0));
  }
  return bounds;
}

void VasilievMShellSortBatcherMergeALL::ShellSortOMP(std::vector<ValType> &vec, std::vector<size_t> &bounds,
                                                     size_t threads) {
  const int chunk_count = static_cast<int>(bounds.size()) - 1;

#pragma omp parallel for default(none) shared(vec, bounds, chunk_count) num_threads(threads) schedule(static)
  for (int chunk = 0; chunk < chunk_count; chunk++) {
    const size_t first = bounds[static_cast<size_t>(chunk)];
    const size_t last = bounds[static_cast<size_t>(chunk) + 1];
    const size_t n = last - first;

    for (size_t gap = n / 2; gap > 0; gap /= 2) {
      for (size_t i = first + gap; i < last; i++) {
        ValType tmp = vec[i];
        size_t j = i;
        while (j >= first + gap && vec[j - gap] > tmp) {
          vec[j] = vec[j - gap];
          j -= gap;
        }
        vec[j] = tmp;
      }
    }
  }
}

void VasilievMShellSortBatcherMergeALL::MergeOne(std::vector<ValType> &vec, std::vector<ValType> &buffer,
                                                 std::vector<size_t> &bounds, size_t size, size_t idx,
                                                 size_t chunk_count) {
  const size_t l = idx * 2 * size;
  const size_t mid = std::min(l + size, chunk_count);
  const size_t r = std::min(l + (2 * size), chunk_count);

  const size_t start = bounds[l];
  const size_t middle = bounds[mid];
  const size_t end = bounds[r];

  if (mid == r) {
    std::copy(vec.begin() + static_cast<std::ptrdiff_t>(start), vec.begin() + static_cast<std::ptrdiff_t>(end),
              buffer.begin() + static_cast<std::ptrdiff_t>(start));
  } else {
    std::vector<ValType> l_vect(vec.begin() + static_cast<std::ptrdiff_t>(start),
                                vec.begin() + static_cast<std::ptrdiff_t>(middle));
    std::vector<ValType> r_vect(vec.begin() + static_cast<std::ptrdiff_t>(middle),
                                vec.begin() + static_cast<std::ptrdiff_t>(end));

    std::vector<ValType> merged = BatcherMerge(l_vect, r_vect);
    for (size_t i = 0; i < merged.size(); i++) {
      buffer[start + i] = merged[i];
    }
  }
}

void VasilievMShellSortBatcherMergeALL::CycleMergeTBB(std::vector<ValType> &vec, std::vector<ValType> &buffer,
                                                      std::vector<size_t> &bounds, size_t size) {
  const size_t chunk_count = bounds.size() - 1;
  const size_t merge_count = (chunk_count + (2 * size) - 1) / (2 * size);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, merge_count), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t idx = range.begin(); idx < range.end(); idx++) {
      MergeOne(vec, buffer, bounds, size, idx, chunk_count);
    }
  });
}

void VasilievMShellSortBatcherMergeALL::CycleMergeSTL(std::vector<ValType> &vec, std::vector<ValType> &buffer,
                                                      std::vector<size_t> &bounds, size_t size, size_t threads) {
  const size_t chunk_count = bounds.size() - 1;
  const size_t merge_count = (chunk_count + (2 * size) - 1) / (2 * size);
  const size_t worker_count = std::min(merge_count, static_cast<size_t>(std::max(size_t{1}, threads)));

  std::vector<std::thread> thread_pool;
  thread_pool.reserve(worker_count);

  const size_t base = merge_count / worker_count;
  const size_t rem = merge_count % worker_count;
  size_t current = 0;

  for (size_t wrk = 0; wrk < worker_count; wrk++) {
    const size_t count = base + (wrk < rem ? 1 : 0);
    const size_t begin = current;
    const size_t end = current + count;
    current = end;

    thread_pool.emplace_back([&, begin, end]() {
      for (size_t idx = begin; idx < end; idx++) {
        MergeOne(vec, buffer, bounds, size, idx, chunk_count);
      }
    });
  }

  for (auto &th : thread_pool) {
    if (th.joinable()) {
      th.join();
    }
  }
}

std::vector<ValType> VasilievMShellSortBatcherMergeALL::BatcherMerge(std::vector<ValType> &l, std::vector<ValType> &r) {
  std::vector<ValType> even_l;
  std::vector<ValType> odd_l;
  std::vector<ValType> even_r;
  std::vector<ValType> odd_r;

  SplitEvenOdd(l, even_l, odd_l);
  SplitEvenOdd(r, even_r, odd_r);

  std::vector<ValType> even = Merge(even_l, even_r);
  std::vector<ValType> odd = Merge(odd_l, odd_r);

  std::vector<ValType> res;
  res.reserve(l.size() + r.size());

  for (size_t i = 0; i < even.size() || i < odd.size(); i++) {
    if (i < even.size()) {
      res.push_back(even[i]);
    }
    if (i < odd.size()) {
      res.push_back(odd[i]);
    }
  }

  for (size_t i = 1; i + 1 < res.size(); i += 2) {
    if (res[i] > res[i + 1]) {
      std::swap(res[i], res[i + 1]);
    }
  }

  return res;
}

void VasilievMShellSortBatcherMergeALL::SplitEvenOdd(std::vector<ValType> &vec, std::vector<ValType> &even,
                                                     std::vector<ValType> &odd) {
  even.reserve(even.size() + (vec.size() / 2) + 1);
  odd.reserve(odd.size() + (vec.size() / 2));

  for (size_t i = 0; i < vec.size(); i += 2) {
    even.push_back(vec[i]);
    if (i + 1 < vec.size()) {
      odd.push_back(vec[i + 1]);
    }
  }
}

std::vector<ValType> VasilievMShellSortBatcherMergeALL::Merge(std::vector<ValType> &a, std::vector<ValType> &b) {
  std::vector<ValType> merged;
  merged.reserve(a.size() + b.size());
  size_t i = 0;
  size_t j = 0;

  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      merged.push_back(a[i++]);
    } else {
      merged.push_back(b[j++]);
    }
  }
  while (i < a.size()) {
    merged.push_back(a[i++]);
  }
  while (j < b.size()) {
    merged.push_back(b[j++]);
  }

  return merged;
}

}  // namespace vasiliev_m_shell_sort_batcher_merge
