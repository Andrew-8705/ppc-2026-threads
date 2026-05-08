#pragma once

#include <cstddef>
#include <vector>

#include "task/include/task.hpp"
#include "vasiliev_m_shell_sort_batcher_merge/common/include/common.hpp"

namespace vasiliev_m_shell_sort_batcher_merge {

class VasilievMShellSortBatcherMergeALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit VasilievMShellSortBatcherMergeALL(const InType &in);
  static std::vector<size_t> ChunkBoundaries(size_t vec_size, size_t threads);

  static void ShellSortOMP(std::vector<ValType> &vec, std::vector<size_t> &bounds, size_t threads);

  static void MergeOne(std::vector<ValType> &vec, std::vector<ValType> &buffer, std::vector<size_t> &bounds,
                       size_t size, size_t idx, size_t chunk_count);

  static void CycleMergeTBB(std::vector<ValType> &vec, std::vector<ValType> &buffer, std::vector<size_t> &bounds,
                            size_t size);

  static void CycleMergeSTL(std::vector<ValType> &vec, std::vector<ValType> &buffer, std::vector<size_t> &bounds,
                            size_t size, size_t threads);

  static std::vector<ValType> BatcherMerge(std::vector<ValType> &l, std::vector<ValType> &r);
  static void SplitEvenOdd(std::vector<ValType> &vec, std::vector<ValType> &even, std::vector<ValType> &odd);
  static std::vector<ValType> Merge(std::vector<ValType> &a, std::vector<ValType> &b);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_shell_sort_batcher_merge
