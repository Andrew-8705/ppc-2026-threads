#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "dorogin_v_bin_img_conv_hull_stl/all/include/ops_all.hpp"
#include "dorogin_v_bin_img_conv_hull_stl/common/include/common.hpp"
#include "dorogin_v_bin_img_conv_hull_stl/omp/include/ops_omp.hpp"
#include "dorogin_v_bin_img_conv_hull_stl/seq/include/ops_seq.hpp"
#include "dorogin_v_bin_img_conv_hull_stl/stl/include/ops_stl.hpp"
#include "dorogin_v_bin_img_conv_hull_stl/tbb/include/ops_tbb.hpp"
#include "util/include/perf_test_util.hpp"

namespace nesterov_a_test_task_threads {

class ExampleRunPerfTestThreads : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_{};

  void SetUp() override {
    constexpr int w = 256;
    constexpr int h = 256;
    std::vector<std::uint8_t> data(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0U);
    for (int y = 24; y < 120; ++y) {
      for (int x = 18; x < 125; ++x) {
        data[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w)) + static_cast<std::size_t>(x)] = 1U;
      }
    }
    for (int y = 130; y < 230; ++y) {
      for (int x = 145; x < 245; ++x) {
        data[(static_cast<std::size_t>(y) * static_cast<std::size_t>(w)) + static_cast<std::size_t>(x)] = 1U;
      }
    }
    input_data_ = InType{.width = w, .height = h, .data = std::move(data)};
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return !output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestThreads, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NesterovATestTaskALL, NesterovATestTaskOMP, NesterovATestTaskSEQ,
                                NesterovATestTaskSTL, NesterovATestTaskTBB>(
        PPC_SETTINGS_dorogin_v_bin_img_conv_hull_stl);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestThreads::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestThreads, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nesterov_a_test_task_threads
