#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

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
    int test_size = 500000;
    input_data_.resize(test_size);
    int seed_val = 42;
    for (int &x : input_data_) {
      x = (seed_val % 2001) - 1000;
      seed_val = (seed_val * 73 + 17) % 2001;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::ranges::is_sorted(output_data);
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
  std::string run_type = ppc::performance::GetStringParamName(std::get<2>(info.param));
  std::string name = run_type + "_" + task_name;
  std::replace_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c); }, '_');
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
