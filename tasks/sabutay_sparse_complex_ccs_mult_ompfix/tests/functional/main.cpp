#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "sabutay_sparse_complex_ccs_mult_ompfix/common/include/common.hpp"
#include "sabutay_sparse_complex_ccs_mult_ompfix/omp/include/ops_omp.hpp"
#include "util/include/func_test_util.hpp"

namespace sabutay_sparse_complex_ccs_mult_ompfix {

constexpr double kZero = 0.0;
constexpr double kValue1 = 1.0;
constexpr double kValue2 = 2.0;
constexpr double kValue3 = 3.0;
constexpr double kValue4 = 4.0;
constexpr double kValue5 = 5.0;
constexpr double kValue6 = 6.0;
constexpr double kValue12 = 12.0;
constexpr double kValue19 = 19.0;

class SabutayARunFuncTestsOmpFix : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static auto PrintTestParam(const TestType &test_param) -> std::string {
    return std::to_string(test_param);
  }

 protected:
  void SetUp() override {
    const TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    CCS &a = std::get<0>(input_data_);
    CCS &b = std::get<1>(input_data_);
    CCS &c = test_result_;

    if (params == 0) {
      a.row_count = 2;
      a.col_count = 3;
      a.col_start = {0, 1, 2, 3};
      a.row_index = {0, 1, 0};
      a.nz = {{kValue1, kZero}, {kValue2, kZero}, {kValue3, kZero}};

      b.row_count = 3;
      b.col_count = 2;
      b.col_start = {0, 2, 3};
      b.row_index = {0, 2, 1};
      b.nz = {{kValue4, kZero}, {kValue5, kZero}, {kValue6, kZero}};

      c.row_count = 2;
      c.col_count = 2;
      c.col_start = {0, 1, 2};
      c.row_index = {0, 1};
      c.nz = {{kValue19, kZero}, {kValue12, kZero}};
    } else if (params == 1) {
      a.row_count = 2;
      a.col_count = 2;
      a.col_start = {0, 1, 2};
      a.row_index = {0, 1};
      a.nz = {{kValue1, kZero}, {kValue2, kZero}};

      b.row_count = 2;
      b.col_count = 2;
      b.col_start = {0, 1, 2};
      b.row_index = {1, 0};
      b.nz = {{kValue3, kZero}, {kValue4, kZero}};

      c.row_count = 2;
      c.col_count = 2;
      c.col_start = {0, 1, 2};
      c.row_index = {1, 0};
      c.nz = {{kValue6, kZero}, {kValue4, kZero}};
    } else {
      a.row_count = 1;
      a.col_count = 1;
      a.col_start = {0, 1};
      a.row_index = {0};
      a.nz = {{kValue2, kValue1}};

      b.row_count = 1;
      b.col_count = 1;
      b.col_start = {0, 1};
      b.row_index = {0};
      b.nz = {{kValue1, kValue1}};

      c.row_count = 1;
      c.col_count = 1;
      c.col_start = {0, 1};
      c.row_index = {0};
      c.nz = {{kValue1, kValue3}};
    }
  }

  bool CheckTestOutputData(OutType &output_data) override {
    constexpr double kEps = 1e-14;
    if (test_result_.row_count != output_data.row_count || test_result_.col_count != output_data.col_count ||
        test_result_.col_start.size() != output_data.col_start.size() ||
        test_result_.row_index.size() != output_data.row_index.size() ||
        test_result_.nz.size() != output_data.nz.size()) {
      return false;
    }

    for (std::size_t i = 0; i < test_result_.col_start.size(); ++i) {
      if (test_result_.col_start[i] != output_data.col_start[i]) {
        return false;
      }
    }

    for (int j = 0; j < test_result_.col_count; ++j) {
      std::vector<std::pair<int, std::complex<double>>> expected_column;
      std::vector<std::pair<int, std::complex<double>>> actual_column;

      const int col_begin = test_result_.col_start[static_cast<std::size_t>(j)];
      const int col_end = test_result_.col_start[static_cast<std::size_t>(j + 1)];
      for (int idx = col_begin; idx < col_end; ++idx) {
        const std::size_t pos = static_cast<std::size_t>(idx);
        expected_column.emplace_back(test_result_.row_index[pos], test_result_.nz[pos]);
        actual_column.emplace_back(output_data.row_index[pos], output_data.nz[pos]);
      }

      auto by_row = [](const auto &x, const auto &y) { return x.first < y.first; };
      std::ranges::sort(expected_column, by_row);
      std::ranges::sort(actual_column, by_row);
      for (std::size_t i = 0; i < expected_column.size(); ++i) {
        if (expected_column[i].first != actual_column[i].first ||
            std::abs(expected_column[i].second - actual_column[i].second) > kEps) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() override {
    return input_data_;
  }

 private:
  InType input_data_{};
  OutType test_result_{};
};

namespace {

TEST_P(SabutayARunFuncTestsOmpFix, FuncCCSTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {0, 1, 2};

const auto kTestTasksList = ppc::util::AddFuncTask<SabutaySparseComplexCcsMultOmpFix, InType>(
    kTestParam, PPC_SETTINGS_sabutay_sparse_complex_ccs_mult_ompfix);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SabutayARunFuncTestsOmpFix::PrintFuncTestName<SabutayARunFuncTestsOmpFix>;

INSTANTIATE_TEST_SUITE_P(RunFuncCCSTest, SabutayARunFuncTestsOmpFix, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace sabutay_sparse_complex_ccs_mult_ompfix
