#include "maslova_u_mult_matr_crs/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "maslova_u_mult_matr_crs/common/include/common.hpp"
#include "util/include/util.hpp"

namespace maslova_u_mult_matr_crs {

MaslovaUMultMatrSTL::MaslovaUMultMatrSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MaslovaUMultMatrSTL::ValidationImpl() {
  const auto &input = GetInput();
  const auto &a = std::get<0>(input);
  const auto &b = std::get<1>(input);
  if (a.cols != b.rows || a.rows <= 0 || b.cols <= 0) {
    return false;
  }
  if (a.row_ptr.size() != static_cast<size_t>(a.rows) + 1) {
    return false;
  }
  if (b.row_ptr.size() != static_cast<size_t>(b.rows) + 1) {
    return false;
  }

  return true;
}

bool MaslovaUMultMatrSTL::PreProcessingImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  auto &c = GetOutput();
  c.rows = a.rows;
  c.cols = b.cols;
  return true;
}

int MaslovaUMultMatrSTL::GetRowNNZ(int i, const CRSMatrix &a, const CRSMatrix &b, std::vector<int> &marker) {
  int row_nnz = 0;
  for (int j = a.row_ptr[i]; j < a.row_ptr[i + 1]; ++j) {
    int col_a = a.col_ind[j];
    for (int k = b.row_ptr[col_a]; k < b.row_ptr[col_a + 1]; ++k) {
      int col_b = b.col_ind[k];
      if (marker[col_b] != i) {
        marker[col_b] = i;
        row_nnz++;
      }
    }
  }
  return row_nnz;
}

void MaslovaUMultMatrSTL::FillRowValues(int i, const CRSMatrix &a, const CRSMatrix &b, CRSMatrix &c,
                                        std::vector<double> &acc, std::vector<int> &marker, std::vector<int> &used) {
  used.clear();
  for (int j = a.row_ptr[i]; j < a.row_ptr[i + 1]; ++j) {
    int col_a = a.col_ind[j];
    double val_a = a.values[j];
    for (int k = b.row_ptr[col_a]; k < b.row_ptr[col_a + 1]; ++k) {
      int col_b = b.col_ind[k];
      if (marker[col_b] != i) {
        marker[col_b] = i;
        used.push_back(col_b);
        acc[col_b] = val_a * b.values[k];
      } else {
        acc[col_b] += val_a * b.values[k];
      }
    }
  }

  if (!used.empty()) {
    std::ranges::sort(used);
    int write_pos = c.row_ptr[i];
    for (int col : used) {
      c.values[write_pos] = acc[col];
      c.col_ind[write_pos] = col;
      write_pos++;
      acc[col] = 0.0;
    }
  }
}

bool MaslovaUMultMatrSTL::RunImpl() {
  const auto &a = std::get<0>(GetInput());
  const auto &b = std::get<1>(GetInput());
  auto &c = GetOutput();

  const int rows_a = a.rows;
  const int cols_b = b.cols;
  c.row_ptr.assign(static_cast<size_t>(rows_a) + 1, 0);

  int num_threads = ppc::util::GetNumThreads();
  if (num_threads <= 0) {
    num_threads = 1;
  }
  if (num_threads > rows_a) {
    num_threads = rows_a;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  auto worker_nnz = [&](int start_row, int end_row) {
    std::vector<int> marker(cols_b, -1);
    for (int i = start_row; i < end_row; ++i) {
      c.row_ptr[i + 1] = GetRowNNZ(i, a, b, marker);
    }
  };

  int chunk = rows_a / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk;
    int end = (t == num_threads - 1) ? rows_a : (t + 1) * chunk;
    if (start < end) {
      threads.emplace_back(worker_nnz, start, end);
    }
  }
  for (auto &t : threads) {
    t.join();
  }
  threads.clear();

  for (int i = 0; i < rows_a; ++i) {
    c.row_ptr[i + 1] += c.row_ptr[i];
  }

  c.values.resize(c.row_ptr[rows_a]);
  c.col_ind.resize(c.row_ptr[rows_a]);

  auto worker_values = [&](int start_row, int end_row) {
    std::vector<double> acc(cols_b, 0.0);
    std::vector<int> marker(cols_b, -1);
    std::vector<int> used;
    used.reserve(cols_b);
    for (int i = start_row; i < end_row; ++i) {
      FillRowValues(i, a, b, c, acc, marker, used);
    }
  };

  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk;
    int end = (t == num_threads - 1) ? rows_a : (t + 1) * chunk;
    if (start < end) {
      threads.emplace_back(worker_values, start, end);
    }
  }
  for (auto &t : threads) {
    t.join();
  }

  return true;
}

bool MaslovaUMultMatrSTL::PostProcessingImpl() {
  return true;
}

}  // namespace maslova_u_mult_matr_crs
