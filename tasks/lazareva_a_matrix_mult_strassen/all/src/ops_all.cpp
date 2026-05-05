#include "lazareva_a_matrix_mult_strassen/all/include/ops_all.hpp"

#include <mpi.h>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <array>
#include <cstddef>
#include <utility>
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
    const int n = GetInput().n;
    if (n <= 0) {
      return false;
    }
    const auto expected = static_cast<size_t>(n) * static_cast<size_t>(n);
    return std::cmp_equal(GetInput().a.size(), expected) && std::cmp_equal(GetInput().b.size(), expected);
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
    const auto padded_size = static_cast<size_t>(padded_n_) * static_cast<size_t>(padded_n_);
    result_.assign(padded_size, 0.0);
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&padded_n_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    const auto padded_size = static_cast<size_t>(padded_n_) * static_cast<size_t>(padded_n_);
    a_.assign(padded_size, 0.0);
    b_.assign(padded_size, 0.0);
    result_.assign(padded_size, 0.0);
  }

  return true;
}

bool LazarevaATestTaskALL::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  result_ = StrassenALL(a_, b_, padded_n_);

  return true;
}

bool LazarevaATestTaskALL::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    auto output = UnpadMatrix(result_, padded_n_, n_);
    GetOutput() = output;

    int output_size = static_cast<int>(output.size());
    MPI_Bcast(&output_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(output.data(), output_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    int output_size = 0;
    MPI_Bcast(&output_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> output(static_cast<size_t>(output_size));
    MPI_Bcast(output.data(), output_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    GetOutput() = output;
  }

  return true;
}

int LazarevaATestTaskALL::NextPowerOfTwo(int n) {
  if (n <= 0) {
    return 1;
  }
  int p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

std::vector<double> LazarevaATestTaskALL::PadMatrix(const std::vector<double> &m, int old_n, int new_n) {
  const auto new_size = static_cast<size_t>(new_n) * static_cast<size_t>(new_n);
  std::vector<double> result(new_size, 0.0);
  for (int i = 0; i < old_n; ++i) {
    for (int j = 0; j < old_n; ++j) {
      const auto dst = (static_cast<ptrdiff_t>(i) * new_n) + j;
      const auto src = (static_cast<ptrdiff_t>(i) * old_n) + j;
      result[static_cast<size_t>(dst)] = m[static_cast<size_t>(src)];
    }
  }
  return result;
}

std::vector<double> LazarevaATestTaskALL::UnpadMatrix(const std::vector<double> &m, int old_n, int new_n) {
  const auto new_size = static_cast<size_t>(new_n) * static_cast<size_t>(new_n);
  std::vector<double> result(new_size);
  for (int i = 0; i < new_n; ++i) {
    for (int j = 0; j < new_n; ++j) {
      const auto dst = (static_cast<ptrdiff_t>(i) * new_n) + j;
      const auto src = (static_cast<ptrdiff_t>(i) * old_n) + j;
      result[static_cast<size_t>(dst)] = m[static_cast<size_t>(src)];
    }
  }
  return result;
}

std::vector<double> LazarevaATestTaskALL::Add(const std::vector<double> &a, const std::vector<double> &b, int n) {
  const auto size = static_cast<size_t>(n) * static_cast<size_t>(n);
  std::vector<double> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

std::vector<double> LazarevaATestTaskALL::Sub(const std::vector<double> &a, const std::vector<double> &b, int n) {
  const auto size = static_cast<size_t>(n) * static_cast<size_t>(n);
  std::vector<double> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

void LazarevaATestTaskALL::Split(const std::vector<double> &parent, int n, std::vector<double> &a11,
                                 std::vector<double> &a12, std::vector<double> &a21, std::vector<double> &a22) {
  const int half = n / 2;
  const auto half_size = static_cast<size_t>(half) * static_cast<size_t>(half);
  a11.resize(half_size);
  a12.resize(half_size);
  a21.resize(half_size);
  a22.resize(half_size);

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      const auto idx = static_cast<size_t>((static_cast<ptrdiff_t>(i) * half) + j);
      a11[idx] = parent[static_cast<size_t>((static_cast<ptrdiff_t>(i) * n) + j)];
      a12[idx] = parent[static_cast<size_t>((static_cast<ptrdiff_t>(i) * n) + j + half)];
      a21[idx] = parent[static_cast<size_t>((static_cast<ptrdiff_t>(i + half) * n) + j)];
      a22[idx] = parent[static_cast<size_t>((static_cast<ptrdiff_t>(i + half) * n) + j + half)];
    }
  }
}

std::vector<double> LazarevaATestTaskALL::Merge(const std::vector<double> &c11, const std::vector<double> &c12,
                                                const std::vector<double> &c21, const std::vector<double> &c22, int n) {
  const int full = n * 2;
  const auto full_size = static_cast<size_t>(full) * static_cast<size_t>(full);
  std::vector<double> result(full_size);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const auto src = static_cast<size_t>((static_cast<ptrdiff_t>(i) * n) + j);
      result[static_cast<size_t>((static_cast<ptrdiff_t>(i) * full) + j)] = c11[src];
      result[static_cast<size_t>((static_cast<ptrdiff_t>(i) * full) + j + n)] = c12[src];
      result[static_cast<size_t>((static_cast<ptrdiff_t>(i + n) * full) + j)] = c21[src];
      result[static_cast<size_t>((static_cast<ptrdiff_t>(i + n) * full) + j + n)] = c22[src];
    }
  }
  return result;
}

std::vector<double> LazarevaATestTaskALL::NaiveMult(const std::vector<double> &a, const std::vector<double> &b, int n) {
  const auto size = static_cast<size_t>(n) * static_cast<size_t>(n);
  std::vector<double> c(size, 0.0);

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, n), [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int i = range.begin(); i < range.end(); ++i) {
      for (int k = 0; k < n; ++k) {
        const double aik = a[static_cast<size_t>((static_cast<ptrdiff_t>(i) * n) + k)];
        for (int j = 0; j < n; ++j) {
          c[static_cast<size_t>((static_cast<ptrdiff_t>(i) * n) + j)] +=
              aik * b[static_cast<size_t>((static_cast<ptrdiff_t>(k) * n) + j)];
        }
      }
    }
  });

  return c;
}

std::vector<double> LazarevaATestTaskALL::StrassenTBB(const std::vector<double> &root_a,
                                                      const std::vector<double> &root_b, int root_n) {
  if (root_n <= 64) {
    return NaiveMult(root_a, root_b, root_n);
  }

  const int half = root_n / 2;

  std::vector<double> a11;
  std::vector<double> a12;
  std::vector<double> a21;
  std::vector<double> a22;
  std::vector<double> b11;
  std::vector<double> b12;
  std::vector<double> b21;
  std::vector<double> b22;
  Split(root_a, root_n, a11, a12, a21, a22);
  Split(root_b, root_n, b11, b12, b21, b22);

  std::array<std::vector<double>, 7> lhs;
  std::array<std::vector<double>, 7> rhs;
  lhs.at(0) = Add(a11, a22, half);
  rhs.at(0) = Add(b11, b22, half);
  lhs.at(1) = Add(a21, a22, half);
  rhs.at(1) = b11;
  lhs.at(2) = a11;
  rhs.at(2) = Sub(b12, b22, half);
  lhs.at(3) = a22;
  rhs.at(3) = Sub(b21, b11, half);
  lhs.at(4) = Add(a11, a12, half);
  rhs.at(4) = b22;
  lhs.at(5) = Sub(a21, a11, half);
  rhs.at(5) = Add(b11, b12, half);
  lhs.at(6) = Sub(a12, a22, half);
  rhs.at(6) = Add(b21, b22, half);

  std::array<std::vector<double>, 7> m;

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, 7), [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int k = range.begin(); k < range.end(); ++k) {
      const auto uk = static_cast<size_t>(k);
      m.at(uk) = NaiveMult(lhs.at(uk), rhs.at(uk), half);
    }
  });

  auto c11 = Add(Sub(Add(m.at(0), m.at(3), half), m.at(4), half), m.at(6), half);
  auto c12 = Add(m.at(2), m.at(4), half);
  auto c21 = Add(m.at(1), m.at(3), half);
  auto c22 = Add(Sub(Add(m.at(0), m.at(2), half), m.at(1), half), m.at(5), half);

  return Merge(c11, c12, c21, c22, half);
}

std::vector<double> LazarevaATestTaskALL::StrassenALL(const std::vector<double> &root_a,
                                                      const std::vector<double> &root_b, int root_n) {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if (root_n <= 128 || world_size == 1) {
    if (rank == 0) {
      return StrassenTBB(root_a, root_b, root_n);
    }
    const auto size = static_cast<size_t>(root_n) * static_cast<size_t>(root_n);
    return std::vector<double>(size, 0.0);
  }

  const int half = root_n / 2;

  std::vector<double> a11, a12, a21, a22;
  std::vector<double> b11, b12, b21, b22;

  if (rank == 0) {
    Split(root_a, root_n, a11, a12, a21, a22);
    Split(root_b, root_n, b11, b12, b21, b22);
  }

  std::array<std::vector<double>, 7> lhs;
  std::array<std::vector<double>, 7> rhs;

  if (rank == 0) {
    lhs.at(0) = Add(a11, a22, half);
    rhs.at(0) = Add(b11, b22, half);
    lhs.at(1) = Add(a21, a22, half);
    rhs.at(1) = b11;
    lhs.at(2) = a11;
    rhs.at(2) = Sub(b12, b22, half);
    lhs.at(3) = a22;
    rhs.at(3) = Sub(b21, b11, half);
    lhs.at(4) = Add(a11, a12, half);
    rhs.at(4) = b22;
    lhs.at(5) = Sub(a21, a11, half);
    rhs.at(5) = Add(b11, b12, half);
    lhs.at(6) = Sub(a12, a22, half);
    rhs.at(6) = Add(b21, b22, half);
  }

  std::array<std::vector<double>, 7> m;
  const auto matrix_size = static_cast<size_t>(half) * static_cast<size_t>(half);

  for (int k = 0; k < 7; ++k) {
    const int target_rank = k % world_size;

    std::vector<double> local_lhs;
    std::vector<double> local_rhs;

    if (rank == 0) {
      if (target_rank == 0) {
        local_lhs = lhs.at(k);
        local_rhs = rhs.at(k);
      } else {
        MPI_Send(lhs.at(k).data(), static_cast<int>(matrix_size), MPI_DOUBLE, target_rank, k * 2, MPI_COMM_WORLD);
        MPI_Send(rhs.at(k).data(), static_cast<int>(matrix_size), MPI_DOUBLE, target_rank, k * 2 + 1, MPI_COMM_WORLD);
      }
    } else if (rank == target_rank) {
      local_lhs.resize(matrix_size);
      local_rhs.resize(matrix_size);
      MPI_Recv(local_lhs.data(), static_cast<int>(matrix_size), MPI_DOUBLE, 0, k * 2, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(local_rhs.data(), static_cast<int>(matrix_size), MPI_DOUBLE, 0, k * 2 + 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    if (rank == target_rank) {
      std::vector<double> result = NaiveMult(local_lhs, local_rhs, half);

      if (rank == 0) {
        m.at(k) = std::move(result);
      } else {
        MPI_Send(result.data(), static_cast<int>(matrix_size), MPI_DOUBLE, 0, k + 100, MPI_COMM_WORLD);
      }
    }

    if (rank == 0 && target_rank != 0) {
      m.at(k).resize(matrix_size);
      MPI_Recv(m.at(k).data(), static_cast<int>(matrix_size), MPI_DOUBLE, target_rank, k + 100, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }

  if (rank == 0) {
    auto c11 = Add(Sub(Add(m.at(0), m.at(3), half), m.at(4), half), m.at(6), half);
    auto c12 = Add(m.at(2), m.at(4), half);
    auto c21 = Add(m.at(1), m.at(3), half);
    auto c22 = Add(Sub(Add(m.at(0), m.at(2), half), m.at(1), half), m.at(5), half);

    return Merge(c11, c12, c21, c22, half);
  }

  const auto size = static_cast<size_t>(root_n) * static_cast<size_t>(root_n);
  return std::vector<double>(size, 0.0);
}

}  // namespace lazareva_a_matrix_mult_strassen
