#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "lazareva_a_matrix_mult_strassen/all/include/ops_all.hpp"
#include "util/include/perf_test_util.hpp"

namespace lazareva_a_matrix_mult_strassen {

class LazarevaARunPerfTestThreads : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kN_ = 512;
  InType input_data_{};
  OutType expected_output_;

  void SetUp() override {
    size_t size = (size_t)kN_ * kN_;
    std::vector<double> a(size), b(size);
    for (size_t i = 0; i < size; ++i) {
      a[i] = (double)((i % 7) + 1);
      b[i] = (double)(((i * 3 + 5) % 11) + 1);
    }
    input_data_ = MatrixInput{.a = a, .b = b, .n = kN_};

    expected_output_.assign(size, 0.0);
    for (int i = 0; i < kN_; ++i) {
      for (int k = 0; k < kN_; ++k) {
        double aik = a[(size_t)i * kN_ + k];
        for (int j = 0; j < kN_; ++j) {
          expected_output_[(size_t)i * kN_ + j] += aik * b[(size_t)k * kN_ + j];
        }
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    constexpr double kEps = 1e-5;
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::abs(output_data[i] - expected_output_[i]) > kEps) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(LazarevaARunPerfTestThreads, RunPerfModes) {
  ExecuteTest(GetParam());
}

INSTANTIATE_TEST_SUITE_P(RunModeTests, LazarevaARunPerfTestThreads,
                         ppc::util::TupleToGTestValues(ppc::util::MakeAllPerfTasks<InType, LazarevaATestTaskALL>(
                             PPC_SETTINGS_lazareva_a_matrix_mult_strassen)),
                         LazarevaARunPerfTestThreads::CustomPerfTestName);

}  // namespace lazareva_a_matrix_mult_strassen
