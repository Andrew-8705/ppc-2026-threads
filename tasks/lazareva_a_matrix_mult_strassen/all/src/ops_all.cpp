#include "lazareva_a_matrix_mult_strassen/all/include/ops_all.hpp"

#include <mpi.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <array>
#include <cstddef>
#include <vector>

#include "lazareva_a_matrix_mult_strassen/common/include/common.hpp"

namespace lazareva_a_matrix_mult_strassen {

int LazarevaATestTaskALL::NextPowerOfTwo(int n) {
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

std::vector<double> LazarevaATestTaskALL::PadMatrix(const std::vector<double> &m, int old_n, int new_n) {
  std::vector<double> r(static_cast<size_t>(new_n) * new_n, 0.0);
  for (int i = 0; i < old_n; ++i) {
    for (int j = 0; j < old_n; ++j) {
      r[i * new_n + j] = m[i * old_n + j];
    }
  }
  return r;
}

std::vector<double> LazarevaATestTaskALL::UnpadMatrix(const std::vector<double> &m, int old_n, int new_n) {
  std::vector<double> r(static_cast<size_t>(new_n) * new_n);
  for (int i = 0; i < new_n; ++i) {
    for (int j = 0; j < new_n; ++j) {
      r[i * new_n + j] = m[i * old_n + j];
    }
  }
  return r;
}

std::vector<double> LazarevaATestTaskALL::Add(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> r(static_cast<size_t>(n) * n);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = a[i] + b[i];
  }
  return r;
}

std::vector<double> LazarevaATestTaskALL::Sub(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> r(static_cast<size_t>(n) * n);
  for (size_t i = 0; i < r.size(); ++i) {
    r[i] = a[i] - b[i];
  }
  return r;
}

void LazarevaATestTaskALL::Split(const std::vector<double> &p, int n, std::vector<double> &a11,
                                 std::vector<double> &a12, std::vector<double> &a21, std::vector<double> &a22) {
  int h = n / 2;
  a11.resize(h * h);
  a12.resize(h * h);
  a21.resize(h * h);
  a22.resize(h * h);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      int idx = i * h + j;
      a11[idx] = p[i * n + j];
      a12[idx] = p[i * n + j + h];
      a21[idx] = p[(i + h) * n + j];
      a22[idx] = p[(i + h) * n + j + h];
    }
  }
}

std::vector<double> LazarevaATestTaskALL::Merge(const std::vector<double> &c11, const std::vector<double> &c12,
                                                const std::vector<double> &c21, const std::vector<double> &c22, int n) {
  int full = n * 2;
  std::vector<double> r(static_cast<size_t>(full) * full);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int idx = i * n + j;
      r[i * full + j] = c11[idx];
      r[i * full + j + n] = c12[idx];
      r[(i + n) * full + j] = c21[idx];
      r[(i + n) * full + j + n] = c22[idx];
    }
  }
  return r;
}

static void NaiveSeq(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int n) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      double aik = a[i * n + k];
      for (int j = 0; j < n; ++j) {
        c[i * n + j] += aik * b[k * n + j];
      }
    }
  }
}

static void NaiveTBB(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int n) {
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, n), [&](const oneapi::tbb::blocked_range<int> &r) {
    for (int i = r.begin(); i < r.end(); ++i) {
      for (int k = 0; k < n; ++k) {
        double aik = a[i * n + k];
        for (int j = 0; j < n; ++j) {
          c[i * n + j] += aik * b[k * n + j];
        }
      }
    }
  });
}

std::vector<double> LazarevaATestTaskALL::NaiveMult(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(static_cast<size_t>(n) * n, 0.0);

  if (n <= 32) {
    NaiveSeq(a, b, c, n);
  } else {
    NaiveTBB(a, b, c, n);
  }

  return c;
}

std::vector<double> LazarevaATestTaskALL::StrassenTBB(const std::vector<double> &a, const std::vector<double> &b,
                                                      int n) {
  if (n <= 128) {
    return NaiveMult(a, b, n);
  }

  int h = n / 2;

  std::vector<double> a11, a12, a21, a22;
  std::vector<double> b11, b12, b21, b22;

  Split(a, n, a11, a12, a21, a22);
  Split(b, n, b11, b12, b21, b22);

  std::array<std::vector<double>, 7> lhs, rhs;

  lhs[0] = Add(a11, a22, h);
  rhs[0] = Add(b11, b22, h);
  lhs[1] = Add(a21, a22, h);
  rhs[1] = b11;
  lhs[2] = a11;
  rhs[2] = Sub(b12, b22, h);
  lhs[3] = a22;
  rhs[3] = Sub(b21, b11, h);
  lhs[4] = Add(a11, a12, h);
  rhs[4] = b22;
  lhs[5] = Sub(a21, a11, h);
  rhs[5] = Add(b11, b12, h);
  lhs[6] = Sub(a12, a22, h);
  rhs[6] = Add(b21, b22, h);

  std::array<std::vector<double>, 7> m;

  oneapi::tbb::parallel_for(0, 7, [&](int k) { m[k] = NaiveMult(lhs[k], rhs[k], h); });

  auto c11 = Add(Sub(Add(m[0], m[3], h), m[4], h), m[6], h);
  auto c12 = Add(m[2], m[4], h);
  auto c21 = Add(m[1], m[3], h);
  auto c22 = Add(Sub(Add(m[0], m[2], h), m[1], h), m[5], h);

  return Merge(c11, c12, c21, c22, h);
}

std::vector<double> LazarevaATestTaskALL::StrassenALL(const std::vector<double> &a, const std::vector<double> &b,
                                                      int n) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (n <= 256 || size == 1) {
    return (rank == 0) ? StrassenTBB(a, b, n) : std::vector<double>(static_cast<size_t>(n) * n, 0.0);
  }

  int h = n / 2;

  std::vector<double> a11, a12, a21, a22;
  std::vector<double> b11, b12, b21, b22;

  if (rank == 0) {
    Split(a, n, a11, a12, a21, a22);
    Split(b, n, b11, b12, b21, b22);
  }

  std::array<std::vector<double>, 7> lhs, rhs;

  if (rank == 0) {
    lhs[0] = Add(a11, a22, h);
    rhs[0] = Add(b11, b22, h);
    lhs[1] = Add(a21, a22, h);
    rhs[1] = b11;
    lhs[2] = a11;
    rhs[2] = Sub(b12, b22, h);
    lhs[3] = a22;
    rhs[3] = Sub(b21, b11, h);
    lhs[4] = Add(a11, a12, h);
    rhs[4] = b22;
    lhs[5] = Sub(a21, a11, h);
    rhs[5] = Add(b11, b12, h);
    lhs[6] = Sub(a12, a22, h);
    rhs[6] = Add(b21, b22, h);
  }

  size_t sz = static_cast<size_t>(h) * h;
  std::array<std::vector<double>, 7> m;

  std::vector<MPI_Request> reqs;

  if (rank == 0) {
    for (int k = 0; k < 7; ++k) {
      int t = k % size;
      if (t == 0) {
        continue;
      }

      reqs.emplace_back();
      MPI_Isend(lhs[k].data(), sz, MPI_DOUBLE, t, k * 2, MPI_COMM_WORLD, &reqs.back());

      reqs.emplace_back();
      MPI_Isend(rhs[k].data(), sz, MPI_DOUBLE, t, k * 2 + 1, MPI_COMM_WORLD, &reqs.back());
    }
  }

  std::vector<int> tasks;
  for (int k = 0; k < 7; ++k) {
    if (k % size == rank) {
      tasks.push_back(k);
    }
  }

  std::vector<std::vector<double>> res(tasks.size());

  for (size_t i = 0; i < tasks.size(); ++i) {
    int k = tasks[i];

    std::vector<double> l(sz), r(sz);

    if (rank == 0) {
      l = lhs[k];
      r = rhs[k];
    } else {
      MPI_Recv(l.data(), sz, MPI_DOUBLE, 0, k * 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(r.data(), sz, MPI_DOUBLE, 0, k * 2 + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    res[i] = NaiveMult(l, r, h);
  }

  for (size_t i = 0; i < tasks.size(); ++i) {
    int k = tasks[i];
    if (rank == 0) {
      m[k] = res[i];
    } else {
      MPI_Send(res[i].data(), sz, MPI_DOUBLE, 0, k + 100, MPI_COMM_WORLD);
    }
  }

  if (rank == 0 && !reqs.empty()) {
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
  }

  if (rank == 0) {
    for (int k = 0; k < 7; ++k) {
      int t = k % size;
      if (t == 0) {
        continue;
      }

      m[k].resize(sz);
      MPI_Recv(m[k].data(), sz, MPI_DOUBLE, t, k + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    auto c11 = Add(Sub(Add(m[0], m[3], h), m[4], h), m[6], h);
    auto c12 = Add(m[2], m[4], h);
    auto c21 = Add(m[1], m[3], h);
    auto c22 = Add(Sub(Add(m[0], m[2], h), m[1], h), m[5], h);

    return Merge(c11, c12, c21, c22, h);
  }

  return std::vector<double>(static_cast<size_t>(n) * n, 0.0);
}

}  // namespace lazareva_a_matrix_mult_strassen
