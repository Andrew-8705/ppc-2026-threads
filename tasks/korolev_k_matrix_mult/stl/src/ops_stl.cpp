#include "korolev_k_matrix_mult/stl/include/ops_stl.hpp"

#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "korolev_k_matrix_mult/common/include/common.hpp"
#include "korolev_k_matrix_mult/common/include/strassen_impl.hpp"

namespace korolev_k_matrix_mult {

KorolevKMatrixMultSTL::KorolevKMatrixMultSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool KorolevKMatrixMultSTL::ValidationImpl() {
  const auto &in = GetInput();
  return in.n > 0 && in.A.size() == in.n * in.n && in.B.size() == in.n * in.n && GetOutput().empty();
}

bool KorolevKMatrixMultSTL::PreProcessingImpl() {
  GetOutput().resize(GetInput().n * GetInput().n);
  return true;
}

bool KorolevKMatrixMultSTL::RunImpl() {
  const auto &in = GetInput();
  size_t n = in.n;
  size_t np2 = strassen_impl::NextPowerOf2(n);

  auto parallel_run = [](std::vector<std::function<void()>> &tasks) {
    std::vector<std::thread> threads;
    threads.reserve(tasks.size());
    for (auto &t : tasks) {
      threads.emplace_back(t);
    }
    for (auto &th : threads) {
      th.join();
    }
  };

  if (np2 == n) {
    strassen_impl::StrassenMultiply(in.A, in.B, GetOutput(), n, parallel_run);
  } else {
    std::vector<double> A_pad(np2 * np2, 0), B_pad(np2 * np2, 0), C_pad(np2 * np2, 0);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        A_pad[i * np2 + j] = in.A[i * n + j];
        B_pad[i * np2 + j] = in.B[i * n + j];
      }
    }
    strassen_impl::StrassenMultiply(A_pad, B_pad, C_pad, np2, parallel_run);
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        GetOutput()[i * n + j] = C_pad[i * np2 + j];
      }
    }
  }
  return true;
}

bool KorolevKMatrixMultSTL::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace korolev_k_matrix_mult
