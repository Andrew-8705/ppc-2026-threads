#include "morozova_s_strassen_multiplication/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "morozova_s_strassen_multiplication/common/include/common.hpp"

namespace morozova_s_strassen_multiplication {

namespace {

Matrix AddMatrixImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = a(i, j) + b(i, j);
    }
  }

  return result;
}

Matrix SubtractMatrixImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result(i, j) = a(i, j) - b(i, j);
    }
  }

  return result;
}

Matrix MultiplyStandardImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(dynamic, 1) shared(a, b, result, n)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += a(i, k) * b(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;
}

void SplitMatrixImpl(const Matrix &m, Matrix &m11, Matrix &m12, Matrix &m21, Matrix &m22) {
  int n = m.size;
  int half = n / 2;

  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      m11(i, j) = m(i, j);
      m12(i, j) = m(i, j + half);
      m21(i, j) = m(i + half, j);
      m22(i, j) = m(i + half, j + half);
    }
  }
}

Matrix MergeMatricesImpl(const Matrix &m11, const Matrix &m12, const Matrix &m21, const Matrix &m22) {
  int half = m11.size;
  int n = 2 * half;
  Matrix result(n);

#pragma omp parallel for default(none) collapse(2) schedule(static) shared(m11, m12, m21, m22, result, half)
  for (int i = 0; i < half; ++i) {
    for (int j = 0; j < half; ++j) {
      result(i, j) = m11(i, j);
      result(i, j + half) = m12(i, j);
      result(i + half, j) = m21(i, j);
      result(i + half, j + half) = m22(i, j);
    }
  }

  return result;
}

Matrix MultiplyStandardParallelImpl(const Matrix &a, const Matrix &b) {
  int n = a.size;
  Matrix result(n);

#pragma omp parallel default(none) shared(a, b, result, n)
  {
#pragma omp for collapse(2) schedule(dynamic, 1)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += a(i, k) * b(k, j);
        }
        result(i, j) = sum;
      }
    }
  }

  return result;
}

}  // namespace

MorozovaSStrassenMultiplicationOMP::MorozovaSStrassenMultiplicationOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType();
}

bool MorozovaSStrassenMultiplicationOMP::ValidationImpl() {
  return true;
}

bool MorozovaSStrassenMultiplicationOMP::PreProcessingImpl() {
  if (GetInput().empty()) {
    valid_data_ = false;
    return true;
  }

  double size_val = GetInput()[0];
  if (size_val <= 0.0) {
    valid_data_ = false;
    return true;
  }

  int n = static_cast<int>(size_val);

  if (GetInput().size() != 1 + (2 * static_cast<size_t>(n) * static_cast<size_t>(n))) {
    valid_data_ = false;
    return true;
  }

  valid_data_ = true;
  n_ = n;

  a_ = Matrix(n_);
  b_ = Matrix(n_);

  int idx = 1;
  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      a_(i, j) = GetInput()[idx++];
    }
  }

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      b_(i, j) = GetInput()[idx++];
    }
  }

  return true;
}

bool MorozovaSStrassenMultiplicationOMP::RunImpl() {
  if (!valid_data_) {
    return true;
  }

  const int leaf_size = 64;

  if (n_ <= leaf_size) {
    c_ = MultiplyStandardParallel(a_, b_);
  } else {
    c_ = MultiplyStrassenParallel(a_, b_, leaf_size, 0);
  }

  return true;
}

bool MorozovaSStrassenMultiplicationOMP::PostProcessingImpl() {
  OutType &output = GetOutput();
  output.clear();

  if (!valid_data_) {
    return true;
  }

  output.push_back(static_cast<double>(n_));

  for (int i = 0; i < n_; ++i) {
    for (int j = 0; j < n_; ++j) {
      output.push_back(c_(i, j));
    }
  }

  return true;
}

Matrix MorozovaSStrassenMultiplicationOMP::AddMatrix(const Matrix &a, const Matrix &b) {
  return AddMatrixImpl(a, b);
}

Matrix MorozovaSStrassenMultiplicationOMP::SubtractMatrix(const Matrix &a, const Matrix &b) {
  return SubtractMatrixImpl(a, b);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStandard(const Matrix &a, const Matrix &b) {
  return MultiplyStandardImpl(a, b);
}

void MorozovaSStrassenMultiplicationOMP::SplitMatrix(const Matrix &m, Matrix &m11, Matrix &m12, Matrix &m21,
                                                     Matrix &m22) {
  SplitMatrixImpl(m, m11, m12, m21, m22);
}

Matrix MorozovaSStrassenMultiplicationOMP::MergeMatrices(const Matrix &m11, const Matrix &m12, const Matrix &m21,
                                                         const Matrix &m22) {
  return MergeMatricesImpl(m11, m12, m21, m22);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStrassen(const Matrix &a, const Matrix &b, int leaf_size) {
  return MultiplyStrassenParallel(a, b, leaf_size, 0);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStrassenParallel(const Matrix &a, const Matrix &b, int leaf_size,
                                                                    int depth) {  // NOLINT(misc-no-recursion)
  int n = a.size;

  if (n <= leaf_size || n % 2 != 0) {
    return MultiplyStandardParallel(a, b);
  }

  int half = n / 2;

  Matrix a11(half);
  Matrix a12(half);
  Matrix a21(half);
  Matrix a22(half);
  Matrix b11(half);
  Matrix b12(half);
  Matrix b21(half);
  Matrix b22(half);

  SplitMatrix(a, a11, a12, a21, a22);
  SplitMatrix(b, b11, b12, b21, b22);

  Matrix p1;
  Matrix p2;
  Matrix p3;
  Matrix p4;
  Matrix p5;
  Matrix p6;
  Matrix p7;

  if (depth < kMaxParallelDepth) {
#pragma omp parallel sections default(none) \
    shared(a11, a12, a21, a22, b11, b12, b21, b22, leaf_size, depth, p1, p2, p3, p4, p5, p6, p7)
    {
#pragma omp section
      p1 = MultiplyStrassenParallel(a11, SubtractMatrix(b12, b22), leaf_size, depth + 1);

#pragma omp section
      p2 = MultiplyStrassenParallel(AddMatrix(a11, a12), b22, leaf_size, depth + 1);

#pragma omp section
      p3 = MultiplyStrassenParallel(AddMatrix(a21, a22), b11, leaf_size, depth + 1);

#pragma omp section
      p4 = MultiplyStrassenParallel(a22, SubtractMatrix(b21, b11), leaf_size, depth + 1);

#pragma omp section
      p5 = MultiplyStrassenParallel(AddMatrix(a11, a22), AddMatrix(b11, b22), leaf_size, depth + 1);

#pragma omp section
      p6 = MultiplyStrassenParallel(SubtractMatrix(a12, a22), AddMatrix(b21, b22), leaf_size, depth + 1);

#pragma omp section
      p7 = MultiplyStrassenParallel(SubtractMatrix(a11, a21), AddMatrix(b11, b12), leaf_size, depth + 1);
    }
  } else {
    p1 = MultiplyStrassenParallel(a11, SubtractMatrix(b12, b22), leaf_size, depth + 1);
    p2 = MultiplyStrassenParallel(AddMatrix(a11, a12), b22, leaf_size, depth + 1);
    p3 = MultiplyStrassenParallel(AddMatrix(a21, a22), b11, leaf_size, depth + 1);
    p4 = MultiplyStrassenParallel(a22, SubtractMatrix(b21, b11), leaf_size, depth + 1);
    p5 = MultiplyStrassenParallel(AddMatrix(a11, a22), AddMatrix(b11, b22), leaf_size, depth + 1);
    p6 = MultiplyStrassenParallel(SubtractMatrix(a12, a22), AddMatrix(b21, b22), leaf_size, depth + 1);
    p7 = MultiplyStrassenParallel(SubtractMatrix(a11, a21), AddMatrix(b11, b12), leaf_size, depth + 1);
  }

  Matrix c11 = AddMatrix(SubtractMatrix(AddMatrix(p5, p4), p2), p6);
  Matrix c12 = AddMatrix(p1, p2);
  Matrix c21 = AddMatrix(p3, p4);
  Matrix c22 = SubtractMatrix(SubtractMatrix(AddMatrix(p5, p1), p3), p7);

  return MergeMatrices(c11, c12, c21, c22);
}

Matrix MorozovaSStrassenMultiplicationOMP::MultiplyStandardParallel(const Matrix &a, const Matrix &b) {
  return MultiplyStandardParallelImpl(a, b);
}

}  // namespace morozova_s_strassen_multiplication
