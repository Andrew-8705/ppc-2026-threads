#include "baldin_a_radix_sort/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <atomic>
#include <numeric>
#include <util/include/util.hpp>
#include <vector>

#include "baldin_a_radix_sort/common/include/common.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace baldin_a_radix_sort {

BaldinARadixSortTBB::BaldinARadixSortTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BaldinARadixSortTBB::ValidationImpl() {
  return true;
}

bool BaldinARadixSortTBB::PreProcessingImpl() {
  return true;
}

bool BaldinARadixSortTBB::RunImpl() {
  return true;
}

bool BaldinARadixSortTBB::PostProcessingImpl() {
  return true;
}

}  // namespace baldin_a_radix_sort
