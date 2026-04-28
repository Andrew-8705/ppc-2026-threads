#include "votincev_d_radixmerge_sort/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "votincev_d_radixmerge_sort/common/include/common.hpp"

namespace votincev_d_radixmerge_sort {

VotincevDRadixMergeSortALL::VotincevDRadixMergeSortALL(InType in) : input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
}

bool VotincevDRadixMergeSortALL::ValidationImpl() {
  return !input_.empty();
}

bool VotincevDRadixMergeSortALL::PreProcessingImpl() {
  return true;
}

void VotincevDRadixMergeSortALL::LocalRadixSort(uint32_t *begin, uint32_t *end) {
  auto n = static_cast<int32_t>(end - begin);
  if (n <= 1) {
    return;
  }

  uint32_t max_val = *std::ranges::max_element(begin, end);
  std::vector<uint32_t> buffer(static_cast<size_t>(n));
  uint32_t *src = begin;
  uint32_t *dst = buffer.data();

  for (int64_t exp = 1; (static_cast<int64_t>(max_val) / exp) > 0; exp *= 10) {
    std::array<int32_t, 10> count{};
    count.fill(0);

    for (int32_t i = 0; i < n; ++i) {
      size_t digit = static_cast<size_t>((src[static_cast<size_t>(i)] / exp) % 10);
      count.at(digit)++;
    }
    for (size_t i = 1; i < 10; ++i) {
      count.at(i) += count.at(i - 1);
    }
    for (int32_t i = n - 1; i >= 0; --i) {
      size_t digit = static_cast<size_t>((src[static_cast<size_t>(i)] / exp) % 10);
      dst[static_cast<size_t>(--count.at(digit))] = src[static_cast<size_t>(i)];
    }
    std::swap(src, dst);
  }

  if (src != begin) {
    std::ranges::copy(src, src + n, begin);
  }
}

void VotincevDRadixMergeSortALL::Merge(const uint32_t *src, uint32_t *dst, int32_t left, int32_t mid, int32_t right) {
  int32_t i = left;
  int32_t j = mid;
  int32_t k = left;

  while (i < mid && j < right) {
    dst[static_cast<size_t>(k++)] = (src[static_cast<size_t>(i)] <= src[static_cast<size_t>(j)])
                                        ? src[static_cast<size_t>(i++)]
                                        : src[static_cast<size_t>(j++)];
  }
  while (i < mid) {
    dst[static_cast<size_t>(k++)] = src[static_cast<size_t>(i++)];
  }
  while (j < right) {
    dst[static_cast<size_t>(k++)] = src[static_cast<size_t>(j++)];
  }
}

void VotincevDRadixMergeSortALL::OmpLocalSortAndMerge(std::vector<uint32_t> &local_data) {
  int32_t local_n = static_cast<int32_t>(local_data.size());
  if (local_n <= 0) {
    return;
  }

  std::vector<uint32_t> temp_buffer(static_cast<size_t>(local_n));

#pragma omp parallel default(none) shared(local_n, local_data, temp_buffer)
  {
    int32_t tid = static_cast<int32_t>(omp_get_thread_num());
    int32_t n_threads = static_cast<int32_t>(omp_get_num_threads());

    int32_t t_items = local_n / n_threads;
    int32_t t_rem = local_n % n_threads;

    // Замена std::min на тернарный оператор для обхода ошибки C3052 (MSVC + default(none))
    int32_t l = (tid * t_items) + (tid < t_rem ? tid : t_rem);
    int32_t r = l + t_items + (tid < t_rem ? 1 : 0);

    if (l < r) {
      LocalRadixSort(local_data.data() + l, local_data.data() + r);
    }

    for (int32_t step = 1; step < n_threads; step *= 2) {
#pragma omp barrier
      if (tid % (2 * step) == 0 && tid + step < n_threads) {
        int32_t next_tid_offset = tid + step;
        int32_t m = (next_tid_offset * t_items) + (next_tid_offset < t_rem ? next_tid_offset : t_rem);

        int32_t double_step_tid = tid + (2 * step);
        int32_t next_tid = (double_step_tid < n_threads ? double_step_tid : n_threads);
        int32_t next_r = (next_tid * t_items) + (next_tid < t_rem ? next_tid : t_rem);

        Merge(local_data.data(), temp_buffer.data(), l, m, next_r);
        std::ranges::copy(temp_buffer.begin() + l, temp_buffer.begin() + next_r, local_data.begin() + l);
      }
    }
  }
}

bool VotincevDRadixMergeSortALL::RunImpl() {
  int32_t rank = 0;
  int32_t size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int32_t n = (rank == 0) ? static_cast<int32_t>(input_.size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

  std::vector<int32_t> send_counts(static_cast<size_t>(size));
  std::vector<int32_t> displacements(static_cast<size_t>(size));
  int32_t items = n / size;
  int32_t rem = n % size;

  for (int32_t i = 0; i < size; ++i) {
    send_counts.at(static_cast<size_t>(i)) = items + (i < rem ? 1 : 0);
    displacements.at(static_cast<size_t>(i)) =
        (i == 0) ? 0 : displacements.at(static_cast<size_t>(i - 1)) + send_counts.at(static_cast<size_t>(i - 1));
  }

  int32_t local_n = send_counts.at(static_cast<size_t>(rank));
  std::vector<uint32_t> local_data(static_cast<size_t>(local_n));

  int32_t min_val = 0;
  if (rank == 0) {
    min_val = *std::ranges::min_element(input_);
  }
  MPI_Bcast(&min_val, 1, MPI_INT32_T, 0, MPI_COMM_WORLD);

  std::vector<uint32_t> unsigned_input;
  if (rank == 0) {
    unsigned_input.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      unsigned_input.at(static_cast<size_t>(i)) = static_cast<uint32_t>(input_.at(static_cast<size_t>(i)) - min_val);
    }
  }

  MPI_Scatterv(unsigned_input.data(), send_counts.data(), displacements.data(), MPI_UINT32_T, local_data.data(),
               local_n, MPI_UINT32_T, 0, MPI_COMM_WORLD);

  OmpLocalSortAndMerge(local_data);

  std::vector<uint32_t> gathered_data;
  if (rank == 0) {
    gathered_data.resize(static_cast<size_t>(n));
  }

  MPI_Gatherv(local_data.data(), local_n, MPI_UINT32_T, gathered_data.data(), send_counts.data(), displacements.data(),
              MPI_UINT32_T, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int32_t i = 1; i < size; ++i) {
      int32_t mid = displacements.at(static_cast<size_t>(i));
      int32_t right = (i == size - 1) ? n : displacements.at(static_cast<size_t>(i + 1));

      std::ranges::inplace_merge(gathered_data.begin(), gathered_data.begin() + mid, gathered_data.begin() + right);
    }

    output_.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      output_.at(static_cast<size_t>(i)) =
          static_cast<int32_t>(gathered_data.at(static_cast<size_t>(i)) + static_cast<uint32_t>(min_val));
    }
  }

  return true;
}

bool VotincevDRadixMergeSortALL::PostProcessingImpl() {
  return true;
}

}  // namespace votincev_d_radixmerge_sort
