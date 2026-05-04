#pragma once

#include <vector>

#include "smyshlaev_a_sle_cg_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace smyshlaev_a_sle_cg_seq {

class SmyshlaevASleCgTaskALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit SmyshlaevASleCgTaskALL(const InType &in);

 private:
  std::vector<double> flat_A_;
  OutType res_;
  int n_ = 0;
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DistributeInitialData(int rank, bool is_mpi, std::vector<double> &b);
  double ComputeDotProductAll(const std::vector<double> &v1, const std::vector<double> &v2, int start, int end,
                              bool is_mpi);
  void ComputeApAll(const std::vector<double> &p, std::vector<double> &ap, int start, int end);
  void SyncVectorP(std::vector<double> &p, int size, bool is_mpi);
  void FinalGather(std::vector<double> &x, int start, int count, int size, bool is_mpi);
};

}  // namespace smyshlaev_a_sle_cg_seq
