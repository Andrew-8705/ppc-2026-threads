#include "popova_e_radix_sort_for_double_with_simple_merge/stl/include/ops_stl.hpp"

#include <atomic>
#include <numeric>
#include <thread>
#include <vector>

#include "popova_e_radix_sort_for_double_with_simple_merge/common/include/common.hpp"
#include "util/include/util.hpp"

namespace popova_e_radix_sort_for_double_with_simple_merge_threads {

PopovaERadixSorForDoubleWithSimpleMergeSTL::PopovaERadixSorForDoubleWithSimpleMergeSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool PopovaERadixSorForDoubleWithSimpleMergeSTL::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool PopovaERadixSorForDoubleWithSimpleMergeSTL::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool PopovaERadixSorForDoubleWithSimpleMergeSTL::RunImpl() {
  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  std::vector<std::thread> threads(num_threads);
  GetOutput() *= num_threads;

  std::atomic<int> counter(0);
  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread([&]() { counter++; });
    threads[i].join();
  }

  GetOutput() /= counter;
  return GetOutput() > 0;
}

bool PopovaERadixSorForDoubleWithSimpleMergeSTL::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace popova_e_radix_sort_for_double_with_simple_merge_threads
