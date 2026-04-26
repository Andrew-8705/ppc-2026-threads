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

KolotukhinAGaussinBlureALL::KolotukhinAGaussinBlureALL(const InType &in)
    : rank_(-1), proc_count_(-1), local_height_(-1) {
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

  int width = img_width;
  int height = img_height;

  MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    global_width_ = width;
    global_height_ = height;
  } else {
    global_width_ = img_width;
    global_height_ = img_height;
  }

  if (rank_ == 0) {
    GetOutput().assign(static_cast<std::size_t>(global_height_) * static_cast<std::size_t>(global_width_), 0);
  }

  DistributeWork();
  return true;
}

void KolotukhinAGaussinBlureALL::DistributeWork() {
  const auto &pixel_data = get<0>(GetInput());

  int rows_per_process = global_height_ / proc_count_;
  int remainder = global_height_ % proc_count_;
  int last_handler = proc_count_ - 1;
  if (rows_per_process == 0) {
    last_handler = remainder - 1;
  }
  local_height_ = (rank_ < remainder) ? rows_per_process + 1 : rows_per_process;

  if (local_height_ == 0) {
    local_data_.clear();
    return;
  }

  local_height_ = (rank_ == 0 || rank_ == last_handler) ? local_height_ + 1 : local_height_ + 2;
  std::size_t local_size = static_cast<std::size_t>(local_height_) * static_cast<std::size_t>(global_width_);
  local_data_.resize(local_size, 0);
  if (rank_ == 0) {
    MPI_Barrier(MPI_COMM_WORLD);
    int current_row = 0;
    for (int dest = 0; dest < proc_count_; dest++) {
      int dest_rows = (dest < remainder) ? rows_per_process + 1 : rows_per_process;
      if (dest_rows == 0) {
        continue;
      }

      int start_row = current_row;
      int end_row = current_row + dest_rows;

      int extended_start = std::max(0, start_row - 1);
      int extended_end = std::min(global_height_, end_row + 1);
      int extended_rows = extended_end - extended_start;

      std::vector<std::uint8_t> extended_data(static_cast<std::size_t>(extended_rows) *
                                              static_cast<std::size_t>(global_width_));

      std::copy(pixel_data.begin() + static_cast<std::size_t>(extended_start) * global_width_,
                pixel_data.begin() + static_cast<std::size_t>(extended_end) * global_width_, extended_data.begin());

      if (dest == 0) {
        local_data_ = std::move(extended_data);
      } else {
        MPI_Send(extended_data.data(), static_cast<int>(extended_data.size()), MPI_UNSIGNED_CHAR, dest, 0,
                 MPI_COMM_WORLD);
      }
      current_row += dest_rows;
    }
  } else {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Recv(local_data_.data(), static_cast<int>(local_data_.size()), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
}

void KolotukhinAGaussinBlureALL::ApplyGaussianBlur(const std::vector<std::uint8_t> &src_data,
                                                   std::vector<std::uint8_t> &dst_data, int width, int height,
                                                   int start_row, int end_row) {
  const static std::array<std::array<int, 3>, 3> kKernel = {{{{1, 2, 1}}, {{2, 4, 2}}, {{1, 2, 1}}}};
  const static int kSum = 16;
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(src_data, dst_data, width, height, start_row, end_row, kKernel, kSum)
  for (int row = start_row; row < end_row; row++) {
    for (int col = 0; col < width; col++) {
      int acc = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          std::uint8_t pixel = GetPixel(src_data, width, height, col + dx, row + dy);
          acc += kKernel.at(1 + dy).at(1 + dx) * static_cast<int>(pixel);
        }
      }
      dst_data[(static_cast<std::size_t>(row) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col)] =
          static_cast<std::uint8_t>(acc / kSum);
    }
  }
}

void KolotukhinAGaussinBlureALL::GatherResults() {
  auto &output = GetOutput();

  int rows_per_process = global_height_ / proc_count_;
  int remainder = global_height_ % proc_count_;

  if (rank_ != 0) {
    int original_rows = (rank_ < remainder) ? rows_per_process + 1 : rows_per_process;

    if (original_rows > 0) {
      std::vector<std::uint8_t> result_only_own(original_rows * global_width_);
      int halo_offset = (rank_ == 0) ? 0 : 1;

      std::copy(local_data_.begin() + halo_offset * global_width_,
                local_data_.begin() + (halo_offset + original_rows) * global_width_, result_only_own.begin());

      MPI_Send(result_only_own.data(), static_cast<int>(result_only_own.size()), MPI_UNSIGNED_CHAR, 0, 1,
               MPI_COMM_WORLD);
    }
  } else {
    std::vector<int> recv_counts(proc_count_);
    std::vector<int> displs(proc_count_);

    int current_row = 0;
    for (int i = 0; i < proc_count_; i++) {
      int rows = (i < remainder) ? rows_per_process + 1 : rows_per_process;
      recv_counts[i] = rows * global_width_;
      displs[i] = current_row * global_width_;
      current_row += rows;
    }

    int root_original_rows = (0 < remainder) ? rows_per_process + 1 : rows_per_process;
    std::copy(local_data_.begin(), local_data_.begin() + root_original_rows * global_width_,
              output.begin() + displs[0]);

    for (int src = 1; src < proc_count_; src++) {
      int src_rows = (src < remainder) ? rows_per_process + 1 : rows_per_process;
      if (src_rows == 0) {
        continue;
      }

      std::vector<std::uint8_t> src_data(src_rows * global_width_);
      MPI_Recv(src_data.data(), static_cast<int>(src_data.size()), MPI_UNSIGNED_CHAR, src, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      std::copy(src_data.begin(), src_data.end(), output.begin() + displs[src]);
    }
  }
}

bool KolotukhinAGaussinBlureALL::RunImpl() {
  if (local_height_ == 0) {
    return true;
  }

  int rows_per_process = global_height_ / proc_count_;
  int remainder = global_height_ % proc_count_;

  int last_handler = proc_count_ - 1;
  if (rows_per_process == 0) {
    last_handler = remainder - 1;
  }

  int extended_height = static_cast<int>(local_data_.size() / global_width_);

  int begin = (rank_ == 0) ? 0 : 1;
  int end = (rank_ == last_handler) ? extended_height : extended_height - 1;

  std::vector<std::uint8_t> result(local_data_.size());
  ApplyGaussianBlur(local_data_, result, global_width_, extended_height, begin, end);

  local_data_ = std::move(result);
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
