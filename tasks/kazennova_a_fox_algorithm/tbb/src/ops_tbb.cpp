#include "kazennova_a_fox_algorithm/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace kazennova_a_fox_algorithm {

KazennovaATestTaskTBB::KazennovaATestTaskTBB(const InType& in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KazennovaATestTaskTBB::ValidationImpl() {
  const auto& in = GetInput();
  if (in.A.data.empty() || in.B.data.empty()) return false;
  if (in.A.rows <= 0 || in.A.cols <= 0 || in.B.rows <= 0 || in.B.cols <= 0) return false;
  if (in.A.cols != in.B.rows) return false;
  return true;
}

bool KazennovaATestTaskTBB::PreProcessingImpl() {
  const auto& in = GetInput();
  auto& out = GetOutput();
  out.rows = in.A.rows;
  out.cols = in.B.cols;
  out.data.assign(static_cast<size_t>(out.rows) * out.cols, 0.0);
  return true;
}

bool KazennovaATestTaskTBB::RunImpl() {
  const auto& in = GetInput();
  auto& out = GetOutput();

  const int M = in.A.rows;
  const int K = in.A.cols;
  const int N = in.B.cols;
  const auto& a = in.A.data;
  const auto& b = in.B.data;
  auto& c = out.data;

  const int BS = BLOCK_SIZE;

  const int blocks_i = (M + BS - 1) / BS;
  const int blocks_j = (N + BS - 1) / BS;
  const int blocks_k = (K + BS - 1) / BS;

  auto get_block = [BS](const std::vector<double>& mat,
                        int rows, int cols,
                        int block_row, int block_col,
                        double* block_buf) {
    int start_row = block_row * BS;
    int start_col = block_col * BS;
    int end_row = std::min(start_row + BS, rows);
    int end_col = std::min(start_col + BS, cols);

    for (int i = 0; i < BS; ++i) {
      for (int j = 0; j < BS; ++j) {
        block_buf[i * BS + j] = 0.0;
      }
    }
    for (int i = start_row; i < end_row; ++i) {
      for (int j = start_col; j < end_col; ++j) {
        block_buf[(i - start_row) * BS + (j - start_col)] = mat[i * cols + j];
      }
    }
  };

  tbb::parallel_for(tbb::blocked_range2d<int>(0, blocks_i, 0, blocks_j),
    [&](const tbb::blocked_range2d<int>& r) {
      std::vector<double> block_a(BS * BS);
      std::vector<double> block_b(BS * BS);

      for (int bi = r.rows().begin(); bi != r.rows().end(); ++bi) {
        for (int bj = r.cols().begin(); bj != r.cols().end(); ++bj) {

          for (int bk = 0; bk < blocks_k; ++bk) {
            get_block(a, M, K, bi, bk, block_a.data());
            get_block(b, K, N, bk, bj, block_b.data());


            for (int i = 0; i < BS; ++i) {
              int global_row = bi * BS + i;
              if (global_row >= M) break;
              for (int j = 0; j < BS; ++j) {
                int global_col = bj * BS + j;
                if (global_col >= N) break;
                double sum = 0.0;
                for (int k = 0; k < BS; ++k) {
                  sum += block_a[i * BS + k] * block_b[k * BS + j];
                }
                c[global_row * N + global_col] += sum;
              }
            }
          }
        }
      }
    }
  );

  return true;
}

bool KazennovaATestTaskTBB::PostProcessingImpl() {
  return !GetOutput().data.empty();
}

}  // namespace kazennova_a_fox_algorithm