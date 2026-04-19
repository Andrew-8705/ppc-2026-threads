#include "baldin_a_radix_sort/stl/include/ops_stl.hpp"

#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

#include "baldin_a_radix_sort/common/include/common.hpp"
#include "util/include/util.hpp"

namespace baldin_a_radix_sort {

BaldinARadixSortSTL::BaldinARadixSortSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BaldinARadixSortSTL::ValidationImpl() {
  return true;
}

bool BaldinARadixSortSTL::PreProcessingImpl() {
  return true;
}

bool BaldinARadixSortSTL::RunImpl() {
  return true;
}

bool BaldinARadixSortSTL::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_radix_sort
