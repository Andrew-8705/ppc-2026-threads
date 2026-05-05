#include "kazennova_a_fox_algorithm/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "kazennova_a_fox_algorithm/common/include/common.hpp"

namespace kazennova_a_fox_algorithm {

namespace {
int ChooseBlockSize(int n) {
  int best = 1;
  int sqrt_n = static_cast<int>(std::sqrt(static_cast<double>(n)));
  for (int bs = sqrt_n; bs >= 1; --bs) {
    if (n % bs == 0) {
      best = bs;
      break;
    }
  }
  return best;
}
}  // namespace

KazennovaATestTaskTBB::KazennovaATestTaskTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KazennovaATestTaskTBB::ValidationImpl() {
  const auto &in = GetInput();
  if (in.A.data.empty() || in.B.data.empty()) {
    return false;
  }
  if (in.A.rows <= 0 || in.A.cols <= 0 || in.B.rows <= 0 || in.B.cols <= 0) {
    return false;
  }
  if (in.A.cols != in.B.rows) {
    return false;
  }
  // Требуем квадратные матрицы для простоты (как в OMP версии)
  if (in.A.rows != in.A.cols || in.B.rows != in.B.cols || in.A.rows != in.B.rows) {
    return false;
  }
  return true;
}

void KazennovaATestTaskTBB::MultiplyBlock(const std::vector<double> &a, const std::vector<double> &b,
                                          std::vector<double> &c, int M, int N, int K, int bi, int bj, int bk) {
  int BS = block_size_;
  int start_row = bi * BS;
  int start_col = bj * BS;
  int start_inner = bk * BS;

  int end_row = std::min(start_row + BS, M);
  int end_col = std::min(start_col + BS, N);
  int end_inner = std::min(start_inner + BS, K);

  for (int i = start_row; i < end_row; ++i) {
    for (int j = start_col; j < end_col; ++j) {
      double sum = 0.0;
      for (int kk = start_inner; kk < end_inner; ++kk) {
        sum += a[i * K + kk] * b[kk * N + j];
      }
      c[i * N + j] += sum;
    }
  }
}

bool KazennovaATestTaskTBB::PreProcessingImpl() {
  const auto &in = GetInput();
  matrix_size_ = in.A.rows;
  GetOutput().rows = matrix_size_;
  GetOutput().cols = matrix_size_;
  GetOutput().data.assign(static_cast<size_t>(matrix_size_) * matrix_size_, 0.0);

  block_size_ = ChooseBlockSize(matrix_size_);
  block_count_ = matrix_size_ / block_size_;
  size_t total_blocks = static_cast<size_t>(block_count_) * block_count_;
  size_t block_elements = static_cast<size_t>(block_size_) * block_size_;

  a_blocks_.assign(total_blocks * block_elements, 0.0);
  b_blocks_.assign(total_blocks * block_elements, 0.0);
  c_blocks_.assign(total_blocks * block_elements, 0.0);

  // Декомпозиция матриц A и B на блоки (как в OMP версии, но вынесем в отдельные функции при необходимости)
  // Здесь для краткости используем прямое копирование
  for (int bi = 0; bi < block_count_; ++bi) {
    for (int bj = 0; bj < block_count_; ++bj) {
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          int src_i = bi * block_size_ + i;
          int src_j = bj * block_size_ + j;
          int dst_idx = ((bi * block_count_ + bj) * block_size_ * block_size_) + i * block_size_ + j;
          if (src_i < matrix_size_ && src_j < matrix_size_) {
            a_blocks_[dst_idx] = in.A.data[src_i * matrix_size_ + src_j];
            b_blocks_[dst_idx] = in.B.data[src_i * matrix_size_ + src_j];
          }
        }
      }
    }
  }

  return true;
}

bool KazennovaATestTaskTBB::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  int M = matrix_size_;
  int N = matrix_size_;
  int K = matrix_size_;

  std::fill(out.data.begin(), out.data.end(), 0.0);

  for (int step = 0; step < block_count_; ++step) {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, block_count_, 0, block_count_),
                      [&](const tbb::blocked_range2d<int> &r) {
      for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
        for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
          int k = (i + step) % block_count_;
          int bi = i, bj = j, bk = k;
          MultiplyBlock(in.A.data, in.B.data, out.data, M, N, K, bi, bj, bk);
        }
      }
    });
  }

  return true;
}

bool KazennovaATestTaskTBB::PostProcessingImpl() {
  return !GetOutput().data.empty();
}

}  // namespace kazennova_a_fox_algorithm
