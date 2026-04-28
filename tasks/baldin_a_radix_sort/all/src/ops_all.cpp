#include "baldin_a_radix_sort/all/include/ops_all.hpp"

#include <mpi.h>

#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

#include "baldin_a_radix_sort/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_radix_sort {

BaldinARadixSortALL::BaldinARadixSortALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BaldinARadixSortALL::ValidationImpl() {
  return true;
}

bool BaldinARadixSortALL::PreProcessingImpl() {
  return true;
}

bool BaldinARadixSortALL::RunImpl() {
  return true;
}

bool BaldinARadixSortALL::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_radix_sort
