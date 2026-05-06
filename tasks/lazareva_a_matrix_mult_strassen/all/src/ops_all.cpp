#include "lazareva_a_matrix_mult_strassen/all/include/ops_all.hpp"

#include <mpi.h>
#include <oneapi/tbb/parallel_for.h>

#include <array>
#include <cstddef>
#include <vector>

#include "lazareva_a_matrix_mult_strassen/common/include/common.hpp"

namespace lazareva_a_matrix_mult_strassen {

LazarevaATestTaskALL::LazarevaATestTaskALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool LazarevaATestTaskALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    int n = GetInput().n;
    if (n <= 0) {
      return false;
    }

    size_t expected = static_cast<size_t>(n) * n;
    return GetInput().a.size() == expected && GetInput().b.size() == expected;
  }
  return true;
}

bool LazarevaATestTaskALL::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    n_ = GetInput().n;
    padded_n_ = NextPowerOfTwo(n_);

    a_ = PadMatrix(GetInput().a, n_, padded_n_);
    b_ = PadMatrix(GetInput().b, n_, padded_n_);

    result_.assign(static_cast<size_t>(padded_n_) * padded_n_, 0.0);
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&padded_n_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    size_t size = static_cast<size_t>(padded_n_) * padded_n_;
    a_.assign(size, 0.0);
    b_.assign(size, 0.0);
    result_.assign(size, 0.0);
  }

  return true;
}

bool LazarevaATestTaskALL::RunImpl() {
  result_ = StrassenALL(a_, b_, padded_n_);
  return true;
}

bool LazarevaATestTaskALL::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto out = UnpadMatrix(result_, padded_n_, n_);
    GetOutput() = out;

    int size = static_cast<int>(out.size());
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(out.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    int size = 0;
    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> out(size);
    MPI_Bcast(out.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    GetOutput() = out;
  }

  return true;
}

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
      r[(i * new_n) + j] = m[(i * old_n) + j];
    }
  }
  return r;
}

std::vector<double> LazarevaATestTaskALL::UnpadMatrix(const std::vector<double> &m, int old_n, int new_n) {
  std::vector<double> r(static_cast<size_t>(new_n) * new_n);

  for (int i = 0; i < new_n; ++i) {
    for (int j = 0; j < new_n; ++j) {
      r[(i * new_n) + j] = m[(i * old_n) + j];
    }
  }
  return r;
}

static std::vector<double> Add(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    r[i] = a[i] + b[i];
  }
  return r;
}

static std::vector<double> Sub(const std::vector<double> &a, const std::vector<double> &b) {
  std::vector<double> r(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    r[i] = a[i] - b[i];
  }
  return r;
}

static void Split(const std::vector<double> &p, std::vector<double> &a11, std::vector<double> &a12,
                  std::vector<double> &a21, std::vector<double> &a22, int n) {
  int h = n / 2;

  a11.resize(static_cast<size_t>(h) * h);
  a12.resize(static_cast<size_t>(h) * h);
  a21.resize(static_cast<size_t>(h) * h);
  a22.resize(static_cast<size_t>(h) * h);

  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < h; ++j) {
      int idx = (i * h) + j;

      a11[idx] = p[(i * n) + j];
      a12[idx] = p[(i * n) + j + h];
      a21[idx] = p[((i + h) * n) + j];
      a22[idx] = p[((i + h) * n) + j + h];
    }
  }
}

static void NaiveSeq(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int n) {
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      double aik = a[(i * n) + k];
      for (int j = 0; j < n; ++j) {
        c[(i * n) + j] += aik * b[(k * n) + j];
      }
    }
  }
}

std::vector<double> LazarevaATestTaskALL::NaiveMult(const std::vector<double> &a, const std::vector<double> &b, int n) {
  std::vector<double> c(static_cast<size_t>(n) * n, 0.0);
  NaiveSeq(a, b, c, n);
  return c;
}

std::vector<double> LazarevaATestTaskALL::Merge(const std::vector<double> &c11, const std::vector<double> &c12,
                                                const std::vector<double> &c21, const std::vector<double> &c22, int n) {
  int full = n * 2;
  std::vector<double> r(static_cast<size_t>(full) * full);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      int idx = (i * n) + j;

      r[(i * full) + j] = c11[idx];
      r[(i * full) + j + n] = c12[idx];
      r[((i + n) * full) + j] = c21[idx];
      r[((i + n) * full) + j + n] = c22[idx];
    }
  }

  return r;
}

std::vector<double> LazarevaATestTaskALL::StrassenALL(const std::vector<double> &a, const std::vector<double> &b,
                                                      int n) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (n <= 128 || size == 1) {
    return (rank == 0) ? NaiveMult(a, b, n) : std::vector<double>(static_cast<size_t>(n) * n, 0.0);
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
    lhs[0] = Add(a11, a22);
    rhs[0] = Add(b11, b22);

    lhs[1] = Add(a21, a22);
    rhs[1] = b11;

    lhs[2] = a11;
    rhs[2] = Sub(b12, b22);

    lhs[3] = a22;
    rhs[3] = Sub(b21, b11);

    lhs[4] = Add(a11, a12);
    rhs[4] = b22;

    lhs[5] = Sub(a21, a11);
    rhs[5] = Add(b11, b12);

    lhs[6] = Sub(a12, a22);
    rhs[6] = Add(b21, b22);
  }

  std::array<std::vector<double>, 7> m;

  for (int k = 0; k < 7; ++k) {
    if (k % size == rank) {
      m[k] = NaiveMult(lhs[k], rhs[k], h);
    }
  }

  if (rank == 0) {
    auto c11 = Add(Sub(Add(m[0], m[3]), m[4]), m[6]);
    auto c12 = Add(m[2], m[4]);
    auto c21 = Add(m[1], m[3]);
    auto c22 = Add(Sub(Add(m[0], m[2]), m[1]), m[5]);

    return Merge(c11, c12, c21, c22, h);
  }

  return std::vector<double>(static_cast<size_t>(n) * n, 0.0);
}

}  // namespace lazareva_a_matrix_mult_strassen
