#pragma once

#include <cstddef>
#include <vector>

#include "kazennova_a_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kazennova_a_fox_algorithm {

class KazennovaATestTaskTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }
  explicit KazennovaATestTaskTBB(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static constexpr int BLOCK_SIZE = 64;

  void MultiplyBlock(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int M, int N,
                     int K, int bi, int bj, int bk);

  int matrix_size_ = 0;
  int block_size_ = BLOCK_SIZE;
  int block_count_ = 0;
  std::vector<double> a_blocks_;
  std::vector<double> b_blocks_;
  std::vector<double> c_blocks_;
};

}  // namespace kazennova_a_fox_algorithm
