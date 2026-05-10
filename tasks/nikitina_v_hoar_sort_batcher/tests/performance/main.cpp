#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <random>
#include <string>

#include "nikitina_v_hoar_sort_batcher/all/include/ops_all.hpp"
#include "nikitina_v_hoar_sort_batcher/common/include/common.hpp"
#include "nikitina_v_hoar_sort_batcher/omp/include/ops_omp.hpp"
#include "nikitina_v_hoar_sort_batcher/seq/include/ops_seq.hpp"
#include "nikitina_v_hoar_sort_batcher/stl/include/ops_stl.hpp"
#include "nikitina_v_hoar_sort_batcher/tbb/include/ops_tbb.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitina_v_hoar_sort_batcher {

class NikitinaVHoarSortBatcherPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int test_size = 50000;
    input_data_.resize(test_size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-100000, 100000);
    for (int &x : input_data_) {
      x = dist(gen);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int mpi_rank = 0;
    if (ppc::util::IsUnderMpirun()) {
      mpi_rank = ppc::util::GetMPIRank();
    }
    if (mpi_rank == 0) {
      if (output_data.size() != input_data_.size()) {
        return false;
      }
      return std::ranges::is_sorted(output_data);
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

std::string CustomPrintPerfTestName(const testing::TestParamInfo<ppc::util::PerfTestParam<InType, OutType>> &info) {
  std::string task_name = std::get<1>(info.param);
  std::string run_type = ppc::performance::GetStringParamName(std::get<2>(info.param));  // NOLINT
  std::string name = run_type + "_" + task_name;
  std::ranges::replace_if(name, [](char c) { return !std::isalnum(c); }, '_');
  return name;
}

TEST_P(NikitinaVHoarSortBatcherPerfTests, RunPerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, HoareSortBatcherSEQ, HoareSortBatcherOMP, HoareSortBatcherTBB,
                                HoareSortBatcherSTL, HoareSortBatcherALL>(PPC_SETTINGS_nikitina_v_hoar_sort_batcher);

INSTANTIATE_TEST_SUITE_P(NikitinaVHoarSortBatcherPerfTests, NikitinaVHoarSortBatcherPerfTests,
                         ppc::util::TupleToGTestValues(kAllPerfTasks), CustomPrintPerfTestName);

}  // namespace
}  // namespace nikitina_v_hoar_sort_batcher
