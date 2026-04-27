#include "otcheskov_s_contrast_lin_stretch/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "otcheskov_s_contrast_lin_stretch/common/include/common.hpp"

namespace otcheskov_s_contrast_lin_stretch {

OtcheskovSContrastLinStretchALL::OtcheskovSContrastLinStretchALL(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput().clear();
  if (rank_ == 0) {
    GetInput() = in;
    GetOutput().resize(GetInput().size());
  } else {
    GetInput().clear();
    GetInput().shrink_to_fit();
  }
  
}

bool OtcheskovSContrastLinStretchALL::ValidationImpl() {
  if (rank_ == 0) {
    is_valid_ = !GetInput().empty();
  }
  MPI_Bcast(&is_valid_, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid_;
}

bool OtcheskovSContrastLinStretchALL::PreProcessingImpl() {
  return true;
}

bool OtcheskovSContrastLinStretchALL::RunImpl() {
  if (!is_valid_) {
    return false;
  }

  const InType& input = GetInput();
  OutType& output = GetOutput();
  size_t global_size = 0;
  if (rank_ == 0) {
    global_size = input.size();
  }
  MPI_Bcast(&global_size, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

  // --- 1. Разделение данных ---
  std::vector<int> counts(size_), displs(size_);
  int base = global_size / size_;
  int rem = global_size % size_;

  for (int i = 0; i < size_; ++i) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = i * base + std::min(i, rem);
  }
  size_t local_size = static_cast<size_t>(counts[rank_]);
  std::vector<uint8_t> local_input(local_size);
  std::vector<uint8_t> local_output(local_size);

  // --- 2. Scatter ---
  MPI_Scatterv(rank_ == 0 ? input.data() : nullptr,
               counts.data(), displs.data(), MPI_UINT8_T,
               local_input.data(), local_size, MPI_UINT8_T,
               0, MPI_COMM_WORLD);

  // --- 3. Локальный min/max (OpenMP) ---
  

  // --- 4. Глобальный min/max ---
  // uint8_t global_min, global_max;

  // MPI_Allreduce(&local_min, &global_min, 1,
  //              MPI_UINT8_T, MPI_MIN, MPI_COMM_WORLD);
  // (&local_max, &global_max, 1,
  //              MPI_UINT8_T, MPI_MAX, MPI_COMM_WORLD);

  // --- 5. Обработка ---


  // --- 6. Сбор ---
  MPI_Gatherv(local_output.data(), local_size, MPI_UINT8_T,
              rank_ == 0 ? output.data() : nullptr,
              counts.data(), displs.data(), MPI_UINT8_T,
              0, MPI_COMM_WORLD);
  return true;
}

bool OtcheskovSContrastLinStretchALL::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_contrast_lin_stretch