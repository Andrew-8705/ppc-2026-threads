#include "kolotukhin_a_gaussian_blur/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kolotukhin_a_gaussian_blur/common/include/common.hpp"

namespace kolotukhin_a_gaussian_blur {

KolotukhinAGaussinBlureALL::KolotukhinAGaussinBlureALL(const InType &in) : rank_(0), proc_count_(1), local_height_(0) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count_);
}

bool KolotukhinAGaussinBlureALL::ValidationImpl() {
  const auto &pixel_data = get<0>(GetInput());
  const auto img_width = get<1>(GetInput());
  const auto img_height = get<2>(GetInput());

  bool valid = static_cast<std::size_t>(img_height) * static_cast<std::size_t>(img_width) == pixel_data.size();

  int local_valid = valid ? 1 : 0;
  int global_valid = 0;
  MPI_Allreduce(&local_valid, &global_valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  return global_valid == 1;
}

bool KolotukhinAGaussinBlureALL::PreProcessingImpl() {
  const auto img_width = get<1>(GetInput());
  const auto img_height = get<2>(GetInput());

  if (rank_ == 0) {
    GetOutput().assign(static_cast<std::size_t>(img_height) * static_cast<std::size_t>(img_width), 0);
  }

  DistributeWork();
  return true;
}

void KolotukhinAGaussinBlureALL::DistributeWork() {
  const auto &pixel_data = get<0>(GetInput());
  const auto img_width = get<1>(GetInput());
  const auto img_height = get<2>(GetInput());

  int rows_per_process = img_height / proc_count_;
  int remainder = img_height % proc_count_;

  local_height_ = (rank_ < remainder) ? rows_per_process + 1 : rows_per_process;

  if (local_height_ == 0) {
    local_data_.clear();
    return;
  }

  int extended_height = local_height_ + 2;
  std::size_t local_size = static_cast<std::size_t>(extended_height) * static_cast<std::size_t>(img_width);
  local_data_.resize(local_size, 0);

  if (rank_ == 0) {
    int current_row = 0;

    for (int dest = 0; dest < proc_count_; dest++) {
      int dest_rows = (dest < remainder) ? rows_per_process + 1 : rows_per_process;

      if (dest_rows == 0) {
        continue;
      }

      int start_row = current_row;
      int end_row = current_row + dest_rows;

      int extended_start = std::max(0, start_row - 1);
      int extended_end = std::min(img_height, end_row + 1);
      int extended_rows = extended_end - extended_start;

      std::vector<std::uint8_t> extended_data(static_cast<std::size_t>(extended_rows) *
                                              static_cast<std::size_t>(img_width));

      for (int row = extended_start; row < extended_end; row++) {
        std::copy(pixel_data.begin() + static_cast<std::size_t>(row) * img_width,
                  pixel_data.begin() + static_cast<std::size_t>(row + 1) * img_width,
                  extended_data.begin() + static_cast<std::size_t>(row - extended_start) * img_width);
      }

      if (dest == 0) {
        local_data_ = std::move(extended_data);
      } else {
        MPI_Send(extended_data.data(), static_cast<int>(extended_data.size()), MPI_UNSIGNED_CHAR, dest, 0,
                 MPI_COMM_WORLD);
      }
      current_row += dest_rows;
    }
  } else {
    MPI_Recv(local_data_.data(), static_cast<int>(local_data_.size()), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void KolotukhinAGaussinBlureALL::ApplyGaussianBlur(std::vector<std::uint8_t> &data, int width, int height,
                                                   int start_row, int end_row) {
  const static std::array<std::array<int, 3>, 3> kKernel = {{{{1, 2, 1}}, {{2, 4, 2}}, {{1, 2, 1}}}};
  const static int kSum = 16;

#pragma omp parallel for collapse(2) schedule(static)
  for (int row = start_row; row < end_row; row++) {
    for (int col = 0; col < width; col++) {
      int acc = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          std::uint8_t pixel = GetPixel(data, width, height, col + dx, row + dy);
          acc += kKernel.at(1 + dy).at(1 + dx) * static_cast<int>(pixel);
        }
      }
      data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(acc / kSum);
    }
  }
}

void KolotukhinAGaussinBlureALL::GatherResults() {
  const auto img_width = get<1>(GetInput());
  const auto img_height = get<2>(GetInput());
  auto &output = GetOutput();

  if (rank_ != 0) {
    if (local_height_ > 0) {
      std::vector<std::uint8_t> result_without_borders(static_cast<std::size_t>(local_height_) *
                                                       static_cast<std::size_t>(img_width));
      for (int r = 0; r < local_height_; r++) {
        std::copy(local_data_.begin() + static_cast<std::size_t>(r + 1) * img_width,
                  local_data_.begin() + static_cast<std::size_t>(r + 2) * img_width,
                  result_without_borders.begin() + static_cast<std::size_t>(r) * img_width);
      }

      MPI_Send(result_without_borders.data(), static_cast<int>(result_without_borders.size()), MPI_UNSIGNED_CHAR, 0, 1,
               MPI_COMM_WORLD);
    }
  } else {
    int rows_per_process = img_height / proc_count_;
    int remainder = img_height % proc_count_;

    std::vector<int> recv_counts(proc_count_);
    std::vector<int> displs(proc_count_);

    int current_row = 0;
    for (int i = 0; i < proc_count_; i++) {
      int rows = (i < remainder) ? rows_per_process + 1 : rows_per_process;
      recv_counts.at(i) = rows * img_width;
      displs.at(i) = current_row * img_width;
      current_row += rows;
    }

    if (local_height_ > 0) {
      for (int row = 0; row < local_height_; row++) {
        std::copy(local_data_.begin() + static_cast<std::size_t>(row + 1) * img_width,
                  local_data_.begin() + static_cast<std::size_t>(row + 2) * img_width,
                  output.begin() + static_cast<std::size_t>(displs[0]) + static_cast<std::size_t>(row) * img_width);
      }
    }

    for (int src = 1; src < proc_count_; ++src) {
      int src_rows = (src < remainder) ? rows_per_process + 1 : rows_per_process;
      if (src_rows == 0) {
        continue;
      }

      std::vector<std::uint8_t> src_data(static_cast<std::size_t>(src_rows) * static_cast<std::size_t>(img_width));

      MPI_Recv(src_data.data(), static_cast<int>(src_data.size()), MPI_UNSIGNED_CHAR, src, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      std::copy(src_data.begin(), src_data.end(), output.begin() + displs.at(src));
    }
  }
}

bool KolotukhinAGaussinBlureALL::RunImpl() {
  if (local_height_ == 0) {
    return true;
  }

  const auto img_width = get<1>(GetInput());
  const auto img_height = get<2>(GetInput());

  int extended_height = static_cast<int>(local_data_.size() / img_width);

  ApplyGaussianBlur(local_data_, img_width, extended_height, 1, extended_height - 1);

  return true;
}

std::uint8_t KolotukhinAGaussinBlureALL::GetPixel(const std::vector<std::uint8_t> &pixel_data, int img_width,
                                                  int img_height, int pos_x, int pos_y) {
  std::size_t x = static_cast<std::size_t>(std::max(0, std::min(pos_x, img_width - 1)));
  std::size_t y = static_cast<std::size_t>(std::max(0, std::min(pos_y, img_height - 1)));
  return pixel_data[(y * static_cast<std::size_t>(img_width)) + x];
}

bool KolotukhinAGaussinBlureALL::PostProcessingImpl() {
  GatherResults();
  return true;
}

}  // namespace kolotukhin_a_gaussian_blur
