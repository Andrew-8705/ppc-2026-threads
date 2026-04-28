#pragma once

#include "baldin_a_radix_sort/common/include/common.hpp"
#include "task/include/task.hpp"

namespace baldin_a_radix_sort {

class BaldinARadixSortALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit BaldinARadixSortALL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace baldin_a_radix_sort
