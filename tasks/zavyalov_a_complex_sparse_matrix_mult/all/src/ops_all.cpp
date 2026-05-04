#include "zavyalov_a_complex_sparse_matrix_mult/all/include/ops_all.hpp"

#include <mpi.h>

#include <map>
#include <vector>

#include "oneapi/tbb/parallel_for.h"
#include "util/include/util.hpp"
#include "zavyalov_a_complex_sparse_matrix_mult/common/include/common.hpp"

namespace zavyalov_a_compl_sparse_matr_mult {

ZavyalovAComplSparseMatrMultALL::ZavyalovAComplSparseMatrMultALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
}

bool ZavyalovAComplSparseMatrMultALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }
  const auto &matr_a = std::get<0>(GetInput());
  const auto &matr_b = std::get<1>(GetInput());
  return matr_a.width == matr_b.height;
}

bool ZavyalovAComplSparseMatrMultALL::PreProcessingImpl() {
  return true;
}

static void BroadcastMatrix(SparseMatrix &m) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t meta[3] = {m.height, m.width, m.val.size()};
  MPI_Bcast(meta, 3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  m.height = meta[0];
  m.width = meta[1];
  size_t count = meta[2];

  m.row_ind.resize(count);
  m.col_ind.resize(count);
  m.val.resize(count);

  MPI_Bcast(m.row_ind.data(), static_cast<int>(count), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(m.col_ind.data(), static_cast<int>(count), MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  std::vector<double> re(count), im(count);
  if (rank == 0) {
    for (size_t i = 0; i < count; ++i) {
      re[i] = m.val[i].re;
      im[i] = m.val[i].im;
    }
  }
  MPI_Bcast(re.data(), static_cast<int>(count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(im.data(), static_cast<int>(count), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    for (size_t i = 0; i < count; ++i) {
      m.val[i] = Complex(re[i], im[i]);
    }
  }
}

std::map<std::pair<size_t, size_t>, Complex> ZavyalovAComplSparseMatrMultALL::ComputeLocalChunk(
    const SparseMatrix &matr_a, const SparseMatrix &matr_b, size_t start, size_t end) {
  int num_threads = ppc::util::GetNumThreads();
  std::vector<std::map<std::pair<size_t, size_t>, Complex>> local_maps(num_threads);

#pragma omp parallel for num_threads(num_threads) schedule(static) default(none) \
    shared(matr_a, matr_b, local_maps, start, end)
  for (size_t i = start; i < end; ++i) {
    int tid = omp_get_thread_num();
    size_t row_a = matr_a.row_ind[i];
    size_t col_a = matr_a.col_ind[i];
    Complex val_a = matr_a.val[i];

    for (size_t j = 0; j < matr_b.Count(); ++j) {
      if (col_a == matr_b.row_ind[j]) {
        local_maps[tid][{row_a, matr_b.col_ind[j]}] += val_a * matr_b.val[j];
      }
    }
  }

  std::map<std::pair<size_t, size_t>, Complex> result;
  for (auto &lm : local_maps) {
    for (auto &[key, value] : lm) {
      result[key] += value;
    }
  }
  return result;
}
bool ZavyalovAComplSparseMatrMultALL::RunImpl() {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  SparseMatrix local_b = (rank == 0) ? std::get<1>(GetInput()) : SparseMatrix{};
  BroadcastMatrix(local_b);

  size_t total = 0;
  size_t a_height = 0;
  size_t a_width = 0;
  if (rank == 0) {
    const auto &ma = std::get<0>(GetInput());
    total = ma.Count();
    a_height = ma.height;
    a_width = ma.width;
  }
  MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_height, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(&a_width, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  int blocksize = static_cast<int>(total) / world_size;
  int leftover = static_cast<int>(total) % world_size;
  std::vector<int> sendcounts(world_size), displs(world_size);
  int offset = 0;
  for (int p = 0; p < world_size; ++p) {
    sendcounts[p] = blocksize + (p < leftover ? 1 : 0);
    displs[p] = offset;
    offset += sendcounts[p];
  }
  int local_count = sendcounts[rank];

  std::vector<size_t> local_rows(local_count), local_cols(local_count);
  std::vector<double> local_re(local_count), local_im(local_count);

  if (rank == 0) {
    const auto &ma = std::get<0>(GetInput());

    std::copy(ma.row_ind.begin(), ma.row_ind.begin() + local_count, local_rows.begin());
    std::copy(ma.col_ind.begin(), ma.col_ind.begin() + local_count, local_cols.begin());
    for (int i = 0; i < local_count; ++i) {
      local_re[i] = ma.val[i].re;
      local_im[i] = ma.val[i].im;
    }

    for (int p = 1; p < world_size; ++p) {
      int cnt = sendcounts[p];
      int dsp = displs[p];

      std::vector<double> re_buf(cnt), im_buf(cnt);
      for (int i = 0; i < cnt; ++i) {
        re_buf[i] = ma.val[dsp + i].re;
        im_buf[i] = ma.val[dsp + i].im;
      }

      MPI_Send(ma.row_ind.data() + dsp, cnt, MPI_UNSIGNED_LONG, p, 0, MPI_COMM_WORLD);
      MPI_Send(ma.col_ind.data() + dsp, cnt, MPI_UNSIGNED_LONG, p, 1, MPI_COMM_WORLD);
      MPI_Send(re_buf.data(), cnt, MPI_DOUBLE, p, 2, MPI_COMM_WORLD);
      MPI_Send(im_buf.data(), cnt, MPI_DOUBLE, p, 3, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_rows.data(), local_count, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_cols.data(), local_count, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_re.data(), local_count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(local_im.data(), local_count, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  SparseMatrix local_a;
  local_a.height = a_height;
  local_a.width = a_width;
  local_a.row_ind = local_rows;
  local_a.col_ind = local_cols;
  local_a.val.resize(local_count);
  for (int i = 0; i < local_count; ++i) {
    local_a.val[i] = Complex(local_re[i], local_im[i]);
  }

  auto local_mp = ComputeLocalChunk(local_a, local_b, 0, static_cast<size_t>(local_count));

  std::vector<size_t> rows, cols;
  std::vector<double> re_vals, im_vals;
  for (const auto &[key, val] : local_mp) {
    rows.push_back(key.first);
    cols.push_back(key.second);
    re_vals.push_back(val.re);
    im_vals.push_back(val.im);
  }

  int res_local_count = static_cast<int>(rows.size());
  std::vector<int> all_counts(world_size);
  MPI_Gather(&res_local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> res_displs(world_size, 0);
  int total_count = 0;
  if (rank == 0) {
    for (int p = 0; p < world_size; ++p) {
      res_displs[p] = total_count;
      total_count += all_counts[p];
    }
  }

  std::vector<size_t> all_rows(total_count), all_cols(total_count);
  std::vector<double> all_re(total_count), all_im(total_count);

  MPI_Gatherv(rows.data(), res_local_count, MPI_UNSIGNED_LONG, all_rows.data(), all_counts.data(), res_displs.data(),
              MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Gatherv(cols.data(), res_local_count, MPI_UNSIGNED_LONG, all_cols.data(), all_counts.data(), res_displs.data(),
              MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Gatherv(re_vals.data(), res_local_count, MPI_DOUBLE, all_re.data(), all_counts.data(), res_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(im_vals.data(), res_local_count, MPI_DOUBLE, all_im.data(), all_counts.data(), res_displs.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::map<std::pair<size_t, size_t>, Complex> final_mp;
    for (int i = 0; i < total_count; ++i) {
      final_mp[{all_rows[i], all_cols[i]}] += Complex(all_re[i], all_im[i]);
    }

    SparseMatrix res;
    res.width = local_b.width;
    res.height = a_height;
    for (const auto &[key, val] : final_mp) {
      res.val.push_back(val);
      res.row_ind.push_back(key.first);
      res.col_ind.push_back(key.second);
    }
    GetOutput() = std::move(res);
  }

  return true;
}

bool ZavyalovAComplSparseMatrMultALL::PostProcessingImpl() {
  return true;
}

}  // namespace zavyalov_a_compl_sparse_matr_mult