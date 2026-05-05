#include "tabalaev_a_matrix_mul_strassen/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <stack>
#include <thread>
#include <utility>
#include <vector>

#include "tabalaev_a_matrix_mul_strassen/common/include/common.hpp"
#include "util/include/util.hpp"

namespace tabalaev_a_matrix_mul_strassen {

static constexpr std::size_t kBaseCaseSize = 128;
static constexpr std::size_t kParallelThreshold = 16384;

namespace {
template <typename fnc>
void RunParallel(std::size_t begin, std::size_t end, std::size_t threshold, const fnc &func) {
  std::size_t total = end - begin;

  if (total < threshold) {
    for (std::size_t i = begin; i < end; ++i) {
      func(i);
    }
    return;
  }

  auto num_threads = static_cast<std::size_t>(ppc::util::GetNumThreads());

  num_threads = std::min(num_threads, total);

  std::vector<std::thread> threads;
  std::size_t chunk_size = total / num_threads;

  for (std::size_t i = 0; i < num_threads; ++i) {
    std::size_t current_start = begin + (i * chunk_size);

    std::size_t current_end = (i == num_threads - 1) ? end : current_start + chunk_size;

    threads.emplace_back([current_start, current_end, &func]() {
      for (std::size_t j = current_start; j < current_end; ++j) {
        func(j);
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}

}  // namespace

TabalaevAMatrixMulStrassenSTL::TabalaevAMatrixMulStrassenSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool TabalaevAMatrixMulStrassenSTL::ValidationImpl() {
  const auto &in = GetInput();
  return in.a_rows > 0 && in.a_cols_b_rows > 0 && in.b_cols > 0 &&
         in.a.size() == static_cast<size_t>(in.a_rows * in.a_cols_b_rows) &&
         in.b.size() == static_cast<size_t>(in.a_cols_b_rows * in.b_cols);
}

bool TabalaevAMatrixMulStrassenSTL::PreProcessingImpl() {
  GetOutput() = {};
  const auto &in = GetInput();

  a_rows_ = in.a_rows;
  a_cols_b_rows_ = in.a_cols_b_rows;
  b_cols_ = in.b_cols;

  std::size_t max_dim = std::max({a_rows_, a_cols_b_rows_, b_cols_});
  padded_n_ = 1;
  while (padded_n_ < max_dim) {
    padded_n_ *= 2;
  }

  padded_a_.assign(padded_n_ * padded_n_, 0.0);
  padded_b_.assign(padded_n_ * padded_n_, 0.0);

  RunParallel(0, a_rows_, kParallelThreshold, [&](std::size_t i) {
    std::size_t i_padded = i * padded_n_;
    std::size_t i_cols = i * a_cols_b_rows_;
    for (std::size_t j = 0; j < a_cols_b_rows_; ++j) {
      padded_a_[i_padded + j] = in.a[i_cols + j];
    }
  });

  RunParallel(0, a_cols_b_rows_, kParallelThreshold, [&](std::size_t i) {
    std::size_t i_padded = i * padded_n_;
    std::size_t i_cols = i * b_cols_;
    for (std::size_t j = 0; j < b_cols_; ++j) {
      padded_b_[i_padded + j] = in.b[i_cols + j];
    }
  });

  return true;
}

bool TabalaevAMatrixMulStrassenSTL::RunImpl() {
  result_c_ = StrassenMultiply(padded_a_, padded_b_, padded_n_);

  auto &out = GetOutput();
  out.assign(a_rows_ * b_cols_, 0.0);

  RunParallel(0, a_rows_, kParallelThreshold, [&](std::size_t i) {
    std::size_t i_cols = i * b_cols_;
    std::size_t i_padded = i * padded_n_;
    for (std::size_t j = 0; j < b_cols_; ++j) {
      out[i_cols + j] = result_c_[i_padded + j];
    }
  });

  return true;
}

bool TabalaevAMatrixMulStrassenSTL::PostProcessingImpl() {
  return true;
}

std::vector<double> TabalaevAMatrixMulStrassenSTL::Add(const std::vector<double> &mat_a,
                                                       const std::vector<double> &mat_b) {
  const std::size_t n = mat_a.size();

  std::vector<double> res(n);

  const double *a_ptr = mat_a.data();
  const double *b_ptr = mat_b.data();

  double *res_ptr = res.data();

  RunParallel(0, n, kParallelThreshold, [&](std::size_t i) { res_ptr[i] = a_ptr[i] + b_ptr[i]; });
  return res;
}

std::vector<double> TabalaevAMatrixMulStrassenSTL::Subtract(const std::vector<double> &mat_a,
                                                            const std::vector<double> &mat_b) {
  const std::size_t n = mat_a.size();

  std::vector<double> res(n);

  const double *a_ptr = mat_a.data();
  const double *b_ptr = mat_b.data();

  double *res_ptr = res.data();

  RunParallel(0, n, kParallelThreshold, [&](std::size_t i) { res_ptr[i] = a_ptr[i] - b_ptr[i]; });

  return res;
}

std::vector<double> TabalaevAMatrixMulStrassenSTL::BaseMultiply(const std::vector<double> &mat_a,
                                                                const std::vector<double> &mat_b, std::size_t n) {
  std::vector<double> res(n * n, 0.0);

  const double *a_ptr = mat_a.data();
  const double *b_ptr = mat_b.data();

  double *res_ptr = res.data();

  RunParallel(0, n, kParallelThreshold, [&](std::size_t i) {
    std::size_t i_n = i * n;
    for (std::size_t k = 0; k < n; ++k) {
      double temp = a_ptr[i_n + k];
      if (temp == 0.0) {
        continue;
      }
      std::size_t k_n = k * n;
      for (std::size_t j = 0; j < n; ++j) {
        res_ptr[i_n + j] += temp * b_ptr[k_n + j];
      }
    }
  });

  return res;
}

void TabalaevAMatrixMulStrassenSTL::SplitMatrix(const std::vector<double> &src, std::size_t n, std::vector<double> &c11,
                                                std::vector<double> &c12, std::vector<double> &c21,
                                                std::vector<double> &c22) {
  std::size_t h = n / 2;
  std::size_t sz = h * h;
  c11.resize(sz);
  c12.resize(sz);
  c21.resize(sz);
  c22.resize(sz);

  const double *src_ptr = src.data();
  double *c11_ptr = c11.data();
  double *c12_ptr = c12.data();
  double *c21_ptr = c21.data();
  double *c22_ptr = c22.data();

  RunParallel(0, h, kParallelThreshold, [&](std::size_t i) {
    std::size_t i_n = i * n;
    std::size_t i_h = i * h;
    std::size_t h_n = h * n;
    for (std::size_t j = 0; j < h; ++j) {
      std::size_t src_idx = i_n + j;
      std::size_t dst_idx = i_h + j;
      c11_ptr[dst_idx] = src_ptr[src_idx];
      c12_ptr[dst_idx] = src_ptr[src_idx + h];
      c21_ptr[dst_idx] = src_ptr[src_idx + h_n];
      c22_ptr[dst_idx] = src_ptr[src_idx + h_n + h];
    }
  });
}

std::vector<double> TabalaevAMatrixMulStrassenSTL::CombineMatrix(const std::vector<double> &c11,
                                                                 const std::vector<double> &c12,
                                                                 const std::vector<double> &c21,
                                                                 const std::vector<double> &c22, std::size_t n) {
  std::size_t h = n / 2;
  std::vector<double> res(n * n);

  const double *c11_ptr = c11.data();
  const double *c12_ptr = c12.data();
  const double *c21_ptr = c21.data();
  const double *c22_ptr = c22.data();
  double *res_ptr = res.data();

  RunParallel(0, h, kParallelThreshold, [&](std::size_t i) {
    std::size_t i_h = i * h;
    std::size_t i_n = i * n;
    std::size_t ih_n = (i + h) * n;
    for (std::size_t j = 0; j < h; ++j) {
      std::size_t src_idx = i_h + j;
      res_ptr[i_n + j] = c11_ptr[src_idx];
      res_ptr[i_n + j + h] = c12_ptr[src_idx];
      res_ptr[ih_n + j] = c21_ptr[src_idx];
      res_ptr[ih_n + j + h] = c22_ptr[src_idx];
    }
  });
  return res;
}

std::vector<double> TabalaevAMatrixMulStrassenSTL::StrassenMultiply(const std::vector<double> &mat_a,
                                                                    const std::vector<double> &mat_b, std::size_t n) {
  std::stack<StrassenFrameSTL> frames;
  std::stack<std::vector<double>> results;

  frames.push({mat_a, mat_b, n, 0});

  while (!frames.empty()) {
    StrassenFrameSTL current = std::move(frames.top());
    frames.pop();

    if (current.n <= kBaseCaseSize) {
      results.push(BaseMultiply(current.mat_a, current.mat_b, current.n));
      continue;
    }

    if (current.stage == 8) {
      std::vector<std::vector<double>> p(7);

      for (int i = 6; i >= 0; --i) {
        p[i] = std::move(results.top());
        results.pop();
      }

      std::size_t h = current.n / 2;
      std::size_t sz = h * h;
      std::vector<double> c11(sz);
      std::vector<double> c12(sz);
      std::vector<double> c21(sz);
      std::vector<double> c22(sz);

      double *c11_ptr = c11.data();
      double *c12_ptr = c12.data();
      double *c21_ptr = c21.data();
      double *c22_ptr = c22.data();

      RunParallel(0, sz, kParallelThreshold, [&](std::size_t i) {
        c11_ptr[i] = p[0][i] + p[3][i] - p[4][i] + p[6][i];
        c12_ptr[i] = p[2][i] + p[4][i];
        c21_ptr[i] = p[1][i] + p[3][i];
        c22_ptr[i] = p[0][i] - p[1][i] + p[2][i] + p[5][i];
      });

      results.push(CombineMatrix(c11, c12, c21, c22, current.n));
    } else {
      std::size_t h = current.n / 2;
      std::vector<double> a11;
      std::vector<double> a12;
      std::vector<double> a21;
      std::vector<double> a22;
      std::vector<double> b11;
      std::vector<double> b12;
      std::vector<double> b21;
      std::vector<double> b22;
      SplitMatrix(current.mat_a, current.n, a11, a12, a21, a22);
      SplitMatrix(current.mat_b, current.n, b11, b12, b21, b22);

      frames.push({{}, {}, current.n, 8});

      frames.push({Subtract(a12, a22), Add(b21, b22), h, 0});
      frames.push({Subtract(a21, a11), Add(b11, b12), h, 0});
      frames.push({Add(a11, a12), b22, h, 0});
      frames.push({a22, Subtract(b21, b11), h, 0});
      frames.push({a11, Subtract(b12, b22), h, 0});
      frames.push({Add(a21, a22), b11, h, 0});
      frames.push({Add(a11, a22), Add(b11, b22), h, 0});
    }
  }

  return std::move(results.top());
}

}  // namespace tabalaev_a_matrix_mul_strassen
