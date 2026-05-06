#include "zyazeva_s_matrix_mult_cannon_alg/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <cmath>
#include <cstddef>

namespace zyazeva_s_matrix_mult_cannon_alg {

namespace {

using AlignedVector = std::vector<double, tbb::cache_aligned_allocator<double>>;

inline size_t BlockIndex(size_t row, size_t col, size_t grid_size) {
  return (row * grid_size) + col;
}

inline size_t BlockOffset(size_t row, size_t col, size_t grid_size, size_t block_area) {
  return BlockIndex(row, col, grid_size) * block_area;
}

size_t FindGridSize(int sz) {
  const auto max_threads = tbb::this_task_arena::max_concurrency();
  const int root = static_cast<int>(std::sqrt(max_threads));

  for (int k = root; k >= 1; --k) {
    if (sz % k == 0) {
      return static_cast<size_t>(k);
    }
  }
  return 1;
}

}  // namespace

ZyazevaSMatrixMultCannonAlgTBB::ZyazevaSMatrixMultCannonAlgTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool ZyazevaSMatrixMultCannonAlgTBB::ValidationImpl() {
  const auto &input = GetInput();
  const size_t sz = std::get<0>(input);

  const auto &m1 = std::get<1>(input);
  const auto &m2 = std::get<2>(input);

  return sz > 0 && m1.size() == sz * sz && m2.size() == sz * sz;
}

bool ZyazevaSMatrixMultCannonAlgTBB::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool ZyazevaSMatrixMultCannonAlgTBB::PostProcessingImpl() {
  return true;
}

void ZyazevaSMatrixMultCannonAlgTBB::MultiplyBlocks(const double *a, const double *b, double *c, int block_size) {
  const int bs = block_size;

  for (int i = 0; i < bs; ++i) {
    const double *a_row = a + static_cast<size_t>(i) * bs;
    double *c_row = c + static_cast<size_t>(i) * bs;

    for (int k = 0; k < bs; ++k) {
      const double a_val = a_row[k];
      const double *b_row = b + static_cast<size_t>(k) * bs;

      for (int j = 0; j < bs; ++j) {
        c_row[j] += a_val * b_row[j];
      }
    }
  }
}

bool ZyazevaSMatrixMultCannonAlgTBB::RunImpl() {
  const int sz = static_cast<int>(std::get<0>(GetInput()));
  const auto &m1 = std::get<1>(GetInput());
  const auto &m2 = std::get<2>(GetInput());

  const size_t grid_size = FindGridSize(sz);
  const size_t bs = static_cast<size_t>(sz) / grid_size;

  const size_t gs = grid_size;
  const size_t block_area = bs * bs;
  const size_t total_blocks = gs * gs;

  AlignedVector A(total_blocks * block_area);
  AlignedVector B(total_blocks * block_area);
  AlignedVector C(total_blocks * block_area, 0.0);

  tbb::parallel_for(size_t(0), total_blocks, [&](size_t id) {
    const size_t bi = id / gs;
    const size_t bj = id % gs;

    const size_t base = BlockOffset(bi, bj, gs, block_area);

    for (size_t i = 0; i < bs; ++i) {
      const size_t gi = bi * bs + i;
      const size_t src = gi * sz + bj * bs;
      const size_t dst = base + i * bs;

      std::copy_n(m1.data() + src, bs, A.data() + dst);
      std::copy_n(m2.data() + src, bs, B.data() + dst);
    }
  });
  std::vector<size_t> map_a(total_blocks), map_b(total_blocks);
  std::vector<size_t> next_a(total_blocks), next_b(total_blocks);

  for (size_t i = 0; i < gs; ++i) {
    for (size_t j = 0; j < gs; ++j) {
      map_a[BlockIndex(i, j, gs)] = BlockIndex(i, (j + i) % gs, gs);
      map_b[BlockIndex(i, j, gs)] = BlockIndex((i + j) % gs, j, gs);
    }
  }
  for (size_t step = 0; step < gs; ++step) {
    tbb::parallel_for(size_t(0), total_blocks, [&](size_t id) {
      const size_t a_idx = map_a[id];
      const size_t b_idx = map_b[id];

      MultiplyBlocks(A.data() + a_idx * block_area, B.data() + b_idx * block_area, C.data() + id * block_area,
                     static_cast<int>(bs));
    });
    if (step + 1 < gs) {
      for (size_t i = 0; i < gs; ++i) {
        for (size_t j = 0; j < gs; ++j) {
          next_a[BlockIndex(i, j, gs)] = map_a[BlockIndex(i, (j + 1) % gs, gs)];

          next_b[BlockIndex(i, j, gs)] = map_b[BlockIndex((i + 1) % gs, j, gs)];
        }
      }

      map_a.swap(next_a);
      map_b.swap(next_b);
    }
  }
  std::vector<double> result(sz * sz);

  tbb::parallel_for(size_t(0), total_blocks, [&](size_t id) {
    const size_t bi = id / gs;
    const size_t bj = id % gs;

    const size_t base = id * block_area;

    for (size_t i = 0; i < bs; ++i) {
      const size_t dst = (bi * bs + i) * sz + bj * bs;
      const size_t src = base + i * bs;

      std::copy_n(C.data() + src, bs, result.data() + dst);
    }
  });

  GetOutput() = std::move(result);
  return true;
}

}  // namespace zyazeva_s_matrix_mult_cannon_alg
