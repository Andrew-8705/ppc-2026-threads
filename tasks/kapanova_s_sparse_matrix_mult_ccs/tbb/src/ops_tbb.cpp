#include "kapanova_s_sparse_matrix_mult_ccs/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "kapanova_s_sparse_matrix_mult_ccs/common/include/common.hpp"

namespace kapanova_s_sparse_matrix_mult_ccs {

KapanovaSSparseMatrixMultCCSTBB::KapanovaSSparseMatrixMultCCSTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool KapanovaSSparseMatrixMultCCSTBB::ValidationImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());

  if (a.cols != b.rows) {
    return false;
  }
  if (a.rows == 0 || a.cols == 0 || b.rows == 0 || b.cols == 0) {
    return false;
  }
  if (a.col_ptrs.size() != static_cast<size_t>(a.cols + 1)) {
    return false;
  }
  if (b.col_ptrs.size() != static_cast<size_t>(b.cols + 1)) {
    return false;
  }
  if (a.col_ptrs[0] != 0 || b.col_ptrs[0] != 0) {
    return false;
  }
  if (a.col_ptrs[a.cols] != a.nnz) {
    return false;
  }
  if (b.col_ptrs[b.cols] != b.nnz) {
    return false;
  }
  if (a.values.size() != static_cast<size_t>(a.nnz) || a.row_indices.size() != static_cast<size_t>(a.nnz)) {
    return false;
  }
  if (b.values.size() != static_cast<size_t>(b.nnz) || b.row_indices.size() != static_cast<size_t>(b.nnz)) {
    return false;
  }

  return true;
}

bool KapanovaSSparseMatrixMultCCSTBB::PreProcessingImpl() {
  return true;
}

bool KapanovaSSparseMatrixMultCCSTBB::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  OutType &c = GetOutput();

  c.rows = a.rows;
  c.cols = b.cols;
  c.col_ptrs.assign(c.cols + 1, 0);
  c.nnz = 0;

  struct ThreadLocalData {
    std::vector<double> accum;
    std::vector<bool> row_mask;
    std::vector<size_t> active_rows;
    std::vector<std::vector<size_t>> col_rows;
    std::vector<std::vector<double>> col_vals;

    explicit ThreadLocalData(size_t rows, size_t cols)
        : accum(rows, 0.0), row_mask(rows, false), col_rows(cols), col_vals(cols) {}
  };

  tbb::enumerable_thread_specific<ThreadLocalData> tls_data([&]() { return ThreadLocalData(a.rows, c.cols); });

  tbb::parallel_for(tbb::blocked_range<size_t>(0, c.cols, 512), [&](const tbb::blocked_range<size_t> &range) {
    auto &local = tls_data.local();

    for (size_t j = range.begin(); j != range.end(); ++j) {
      for (size_t k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; ++k) {
        size_t row_b = b.row_indices[k];
        double val_b = b.values[k];

        for (size_t zc = a.col_ptrs[row_b]; zc < a.col_ptrs[row_b + 1]; ++zc) {
          size_t i = a.row_indices[zc];
          double val_a = a.values[zc];

          local.accum[i] += val_a * val_b;
          if (!local.row_mask[i]) {
            local.row_mask[i] = true;
            local.active_rows.push_back(i);
          }
        }
      }

      std::sort(local.active_rows.begin(), local.active_rows.end());

      for (size_t i : local.active_rows) {
        if (local.accum[i] != 0.0) {
          local.col_rows[j].push_back(i);
          local.col_vals[j].push_back(local.accum[i]);
        }
        local.accum[i] = 0.0;
        local.row_mask[i] = false;
      }
      local.active_rows.clear();
    }
  });

  std::vector<size_t> col_sizes(c.cols, 0);
  for (const auto &local : tls_data) {
    for (size_t j = 0; j < c.cols; ++j) {
      col_sizes[j] += local.col_rows[j].size();
    }
  }

  size_t offset = 0;
  for (size_t j = 0; j < c.cols; ++j) {
    c.col_ptrs[j] = offset;
    offset += col_sizes[j];
  }
  c.col_ptrs[c.cols] = offset;
  c.nnz = offset;

  c.values.resize(c.nnz);
  c.row_indices.resize(c.nnz);

  std::vector<size_t> current_pos(c.cols, 0);
  for (const auto &local : tls_data) {
    for (size_t j = 0; j < c.cols; ++j) {
      size_t start = c.col_ptrs[j] + current_pos[j];
      const auto &rows = local.col_rows[j];
      const auto &vals = local.col_vals[j];
      for (size_t idx = 0; idx < rows.size(); ++idx) {
        size_t pos = start + idx;
        c.row_indices[pos] = rows[idx];
        c.values[pos] = vals[idx];
      }
      current_pos[j] += rows.size();
    }
  }

  return true;
}

bool KapanovaSSparseMatrixMultCCSTBB::PostProcessingImpl() {
  return true;
}

}  // namespace kapanova_s_sparse_matrix_mult_ccs
