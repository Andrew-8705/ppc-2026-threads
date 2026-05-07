#include "kazennova_a_fox_algorithm/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <vector>

#include "kazennova_a_fox_algorithm/common/include/common.hpp"

namespace kazennova_a_fox_algorithm {

namespace {

// Вспомогательная функция для извлечения блока (уменьшает когнитивную сложность)
void GetBlock(const std::vector<double> &mat, int rows, int cols, int block_row, int block_col, double *block_buf) {
  const int bs = kBlockSize;
  const int start_row = block_row * bs;
  const int start_col = block_col * bs;
  const int end_row = std::min(start_row + bs, rows);
  const int end_col = std::min(start_col + bs, cols);

  // Обнуление буфера
  for (int i = 0; i < bs; ++i) {
    for (int j = 0; j < bs; ++j) {
      block_buf[i * bs + j] = 0.0;
    }
  }
  // Копирование блока
  for (int i = start_row; i < end_row; ++i) {
    for (int j = start_col; j < end_col; ++j) {
      block_buf[(i - start_row) * bs + (j - start_col)] = mat[i * cols + j];
    }
  }
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
  return true;
}

bool KazennovaATestTaskTBB::PreProcessingImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();
  out.rows = in.A.rows;
  out.cols = in.B.cols;
  out.data.assign(static_cast<size_t>(out.rows) * out.cols, 0.0);
  return true;
}

bool KazennovaATestTaskTBB::RunImpl() {
  const auto &in = GetInput();
  auto &out = GetOutput();

  const int m = in.A.rows;
  const int k = in.A.cols;
  const int n = in.B.cols;
  const auto &a = in.A.data;
  const auto &b = in.B.data;
  auto &c = out.data;

  const int bs = kBlockSize;

  const int blocks_i = (m + bs - 1) / bs;
  const int blocks_j = (n + bs - 1) / bs;
  const int blocks_k = (k + bs - 1) / bs;

  tbb::parallel_for(tbb::blocked_range2d<int>(0, blocks_i, 0, blocks_j), [&](const tbb::blocked_range2d<int> &r) {
    std::vector<double> block_a(static_cast<size_t>(bs) * bs);
    std::vector<double> block_b(static_cast<size_t>(bs) * bs);

    for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
      for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {
        for (int bk = 0; bk < blocks_k; ++bk) {
          GetBlock(a, m, k, bi, bk, block_a.data());
          GetBlock(b, k, n, bk, bj, block_b.data());

          const int max_i = std::min(bs, m - bi * bs);
          const int max_j = std::min(bs, n - bj * bs);
          const int max_k = std::min(bs, k - bk * bs);

          for (int i = 0; i < max_i; ++i) {
            const int global_row = bi * bs + i;
            for (int j = 0; j < max_j; ++j) {
              const int global_col = bj * bs + j;
              double sum = 0.0;
              for (int kk = 0; kk < max_k; ++kk) {
                sum += block_a[i * bs + kk] * block_b[kk * bs + j];
              }
              c[global_row * n + global_col] += sum;
            }
          }
        }
      }
    }
  });

  return true;
}

bool KazennovaATestTaskTBB::PostProcessingImpl() {
  return !GetOutput().data.empty();
}

}  // namespace kazennova_a_fox_algorithm
