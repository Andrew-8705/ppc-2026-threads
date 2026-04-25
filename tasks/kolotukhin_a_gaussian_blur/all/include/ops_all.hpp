#pragma once

#include <cstdint>
#include <vector>

#include "kolotukhin_a_gaussian_blur/common/include/common.hpp"
#include "task/include/task.hpp"

namespace kolotukhin_a_gaussian_blur {

class KolotukhinAGaussinBlureALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }
  explicit KolotukhinAGaussinBlureALL(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] static std::uint8_t GetPixel(const std::vector<std::uint8_t> &pixel_data, int img_width, int img_height,
                                             int pos_x, int pos_y);

  int rank_;
  int proc_count_;
  std::vector<std::uint8_t> local_data_;
  int local_height_;
  int global_height_;
  int global_width_;
  void DistributeWork();
  void GatherResults();
  void ApplyGaussianBlur(const std::vector<std::uint8_t> &src_data, std::vector<std::uint8_t> &dst_data, int width,
                         int height, int start_row, int end_row);
};

}  // namespace kolotukhin_a_gaussian_blur
