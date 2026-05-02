#include "zavyalov_a_complex_sparse_matrix_mult/all/include/ops_all.hpp"

#include <mpi.h>

#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

#include "zavyalov_a_complex_sparse_matrix_mult/common/include/common.hpp"
#include "oneapi/tbb/parallel_for.h"
#include "util/include/util.hpp"

namespace zavyalov_a_compl_sparse_matr_mult {

ZavyalovAComplSparseMatrMultALL::ZavyalovAComplSparseMatrMultALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ZavyalovAComplSparseMatrMultALL::ValidationImpl() {
  const auto &matr_a = std::get<0>(GetInput());
  const auto &matr_b = std::get<1>(GetInput());
  return matr_a.width == matr_b.height;
}

bool ZavyalovAComplSparseMatrMultALL::PreProcessingImpl() {
  return true;
}

bool ZavyalovAComplSparseMatrMultALL::RunImpl() {
  const auto &matr_a = std::get<0>(GetInput());
  const auto &matr_b = std::get<1>(GetInput());

  GetOutput() = MultiplicateWithStl(matr_a, matr_b);

  return true;
}

bool ZavyalovAComplSparseMatrMultALL::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_compl_sparse_matr_mult
