#include "zyazeva_s_matrix_mult_cannon_alg/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace zyazeva_s_matrix_mult_cannon_alg {

namespace {

using AlignedVector = std::vector<double, tbb::cache_aligned_allocator<double>>;

inline size_t BlockIndex(size_t row, size_t col, size_t grid_size) {
  return (row * grid_size) + col;
}

inline size_t BlockOffset(size_t row, size_t col, size_t grid_size, size_t block_area) {
  return BlockIndex(row, col, grid_size) * block_area;
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

  const int max_threads = static_cast<int>(tbb::this_task_arena::max_concurrency());

  int grid_size = 1;
  const int sqrt_threads = static_cast<int>(std::sqrt(max_threads));

  for (int k = sqrt_threads; k >= 1; --k) {
    if (sz % k == 0) {
      grid_size = k;
      break;
    }
  }

  const int block_size = sz / grid_size;

  const size_t gs = static_cast<size_t>(grid_size);
  const size_t bs = static_cast<size_t>(block_size);

  const size_t matrix_size = static_cast<size_t>(sz) * static_cast<size_t>(sz);

  const size_t block_area = bs * bs;
  const size_t total_blocks = gs * gs;

  AlignedVector blocks_a(total_blocks * block_area);
  AlignedVector blocks_b(total_blocks * block_area);
  AlignedVector blocks_c(total_blocks * block_area, 0.0);

  tbb::parallel_for(static_cast<size_t>(0), total_blocks, [&](size_t block_id) {
    const size_t bi = block_id / gs;
    const size_t bj = block_id % gs;

    const size_t a_offset = BlockOffset(bi, bj, gs, block_area);

    for (size_t i = 0; i < bs; ++i) {
      const size_t global_i = bi * bs + i;

      const size_t src_offset = global_i * static_cast<size_t>(sz) + bj * bs;

      const size_t local_offset = a_offset + i * bs;

      std::copy_n(m1.data() + src_offset, bs, blocks_a.data() + local_offset);

      std::copy_n(m2.data() + src_offset, bs, blocks_b.data() + local_offset);
    }
  });

  std::vector<size_t> map_a(total_blocks);
  std::vector<size_t> map_b(total_blocks);

  for (size_t i = 0; i < gs; ++i) {
    for (size_t j = 0; j < gs; ++j) {
      map_a[BlockIndex(i, j, gs)] = BlockIndex(i, (j + i) % gs, gs);

      map_b[BlockIndex(i, j, gs)] = BlockIndex((i + j) % gs, j, gs);
    }
  }

  std::vector<size_t> next_a(total_blocks);
  std::vector<size_t> next_b(total_blocks);

  for (size_t step = 0; step < gs; ++step) {
    tbb::parallel_for(static_cast<size_t>(0), total_blocks, [&](size_t idx) {
      const size_t a_idx = map_a[idx];
      const size_t b_idx = map_b[idx];

      const double *a_ptr = blocks_a.data() + a_idx * block_area;

      const double *b_ptr = blocks_b.data() + b_idx * block_area;

      double *c_ptr = blocks_c.data() + idx * block_area;

      MultiplyBlocks(a_ptr, b_ptr, c_ptr, block_size);
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

  std::vector<double> result(matrix_size);

  tbb::parallel_for(static_cast<size_t>(0), total_blocks, [&](size_t block_id) {
    const size_t bi = block_id / gs;
    const size_t bj = block_id % gs;

    const size_t block_offset = block_id * block_area;

    for (size_t i = 0; i < bs; ++i) {
      const size_t dst_row = (bi * bs + i) * static_cast<size_t>(sz);

      const size_t src_row = block_offset + i * bs;

      std::copy_n(blocks_c.data() + src_row, bs, result.data() + dst_row + bj * bs);
    }
  });

  GetOutput() = std::move(result);
  return true;
}

}  // namespace zyazeva_s_matrix_mult_cannon_alg
