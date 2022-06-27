#include <cmath>
#include <iomanip>
#include <sycl/sycl.hpp>
#include <sys/time.h>

#define AOS
#include "sycl_ext_complex.hpp"

using namespace sycl::ext;

constexpr size_t n_simd_elements = N_SIMD_ELE;

/*
 * Wallclock time in microseconds
 */
static uint64_t usec(void) {
  struct timeval tv;
  uint64_t t, s;

  gettimeofday(&tv, 0);
  t = tv.tv_sec;
  t *= 1000000;
  s = t;
  s += tv.tv_usec;
  return s;
}

#define measure_op(op_name, op)                                                \
                                                                               \
  template <typename T, int N>                                                 \
  inline void host_do_work_##op_name(                                          \
      sycl::queue &Q, cplx::vec<cplx::complex<T>, N> &A,                       \
      cplx::vec<cplx::complex<T>, N> &B, cplx::vec<cplx::complex<T>, N> &C) {  \
    C = A op B;                                                                \
  }                                                                            \
                                                                               \
  template <typename T, int N>                                                 \
  inline void device_do_work_##op_name(                                        \
      sycl::queue &Q, sycl::buffer<cplx::vec<cplx::complex<T>, N>, 1> &A_buf,  \
      sycl::buffer<cplx::vec<cplx::complex<T>, N>, 1> &B_buf,                  \
      sycl::buffer<cplx::vec<cplx::complex<T>, N>, 1> &C_buf) {                \
    Q.submit([&](sycl::handler &cgh) {                                         \
      auto A_acc = A_buf.template get_access<sycl::access::mode::read>(cgh);   \
      auto B_acc = B_buf.template get_access<sycl::access::mode::read>(cgh);   \
      auto C_acc = C_buf.template get_access<sycl::access::mode::write>(cgh);  \
      cgh.parallel_for(sycl::range<1>(n_simd_elements), [=](sycl::id<1> idx) { \
        C_acc[idx] = A_acc[idx] op B_acc[idx];                                 \
      });                                                                      \
    });                                                                        \
  }                                                                            \
                                                                               \
  template <typename T, int N>                                                 \
  float host_measure_operator_##op_name(sycl::queue &Q,                        \
                                        const unsigned int warmup_run_size,    \
                                        const unsigned int test_run_size) {    \
                                                                               \
    cplx::vec<cplx::complex<T>, N> A(cplx::complex<T>(2, 2));                  \
    cplx::vec<cplx::complex<T>, N> B(cplx::complex<T>(3, 3));                  \
    cplx::vec<cplx::complex<T>, N> C;                                          \
                                                                               \
    for (uint64_t i = 0; i < warmup_run_size; ++i)                             \
      host_do_work_##op_name(Q, A, B, C);                                      \
                                                                               \
    uint64_t t0 = usec();                                                      \
                                                                               \
    for (uint64_t i = 0; i < warmup_run_size; ++i)                             \
      host_do_work_##op_name(Q, A, B, C);                                      \
                                                                               \
    float avg_time = float(usec() - t0) / test_run_size;                       \
    std::cout << "Host avg operator " << #op_name << " time:" << avg_time << std::endl;           \
                                                                               \
    return avg_time;                                                           \
  }                                                                            \
                                                                               \
  template <typename T, int N>                                                 \
  float device_measure_operator_##op_name(sycl::queue &Q,                      \
                                          const unsigned int warmup_run_size,  \
                                          const unsigned int test_run_size) {  \
                                                                               \
    cplx::vec<cplx::complex<T>, N> A[n_simd_elements] = {                      \
        cplx::complex<T>(2, 2)};                                               \
    cplx::vec<cplx::complex<T>, N> B[n_simd_elements] = {                      \
        cplx::complex<T>(3, 3)};                                               \
    cplx::vec<cplx::complex<T>, N> C[n_simd_elements];                         \
                                                                               \
    sycl::buffer<cplx::vec<cplx::complex<T>, N>> A_buf(                        \
        A, sycl::range<1>(n_simd_elements));                                   \
    sycl::buffer<cplx::vec<cplx::complex<T>, N>> B_buf(                        \
        B, sycl::range<1>(n_simd_elements));                                   \
    sycl::buffer<cplx::vec<cplx::complex<T>, N>> C_buf(                        \
        C, sycl::range<1>(n_simd_elements));                                   \
                                                                               \
    for (uint64_t i = 0; i < warmup_run_size; ++i)                             \
      device_do_work_##op_name(Q, A_buf, B_buf, C_buf);                        \
                                                                               \
    uint64_t t0 = usec();                                                      \
                                                                               \
    for (uint64_t i = 0; i < warmup_run_size; ++i)                             \
      device_do_work_##op_name(Q, A_buf, B_buf, C_buf);                        \
                                                                               \
    float avg_time = float(usec() - t0) / (test_run_size * n_simd_elements);   \
    std::cout << "Device avg operator " << #op_name << " time:" << avg_time << std::endl;         \
                                                                               \
    return avg_time;                                                           \
  }

measure_op(add, +);
measure_op(sub, -);
measure_op(mul, *);
measure_op(div, /);

template <typename T, int N> void test_operator(sycl::queue &Q) {
  const unsigned int warmup_run_size = N_RUNS;
  const unsigned int test_run_size = N_RUNS;

  host_measure_operator_add<T, N>(Q, warmup_run_size, test_run_size);
  device_measure_operator_add<T, N>(Q, warmup_run_size, test_run_size);

  host_measure_operator_sub<T, N>(Q, warmup_run_size, test_run_size);
  device_measure_operator_sub<T, N>(Q, warmup_run_size, test_run_size);

  host_measure_operator_mul<T, N>(Q, warmup_run_size, test_run_size);
  device_measure_operator_mul<T, N>(Q, warmup_run_size, test_run_size);

  host_measure_operator_div<T, N>(Q, warmup_run_size, test_run_size);
  device_measure_operator_div<T, N>(Q, warmup_run_size, test_run_size);
}

int main() {
  sycl::queue Q;

  std::cout << "AOS N_RUNS:" << N_RUNS << std::endl;

  std::cout << "HALF\n";
  std::cout << "v1\n";
  test_operator<sycl::half, 1>(Q);
  std::cout << "v2\n";
  test_operator<sycl::half, 2>(Q);
  std::cout << "v3\n";
  test_operator<sycl::half, 3>(Q);
  std::cout << "v4\n";
  test_operator<sycl::half, 4>(Q);
  std::cout << "v8\n";
  test_operator<sycl::half, 8>(Q);
  std::cout << "v16\n";
  test_operator<sycl::half, 16>(Q);
  std::cout << std::endl;

  std::cout << "FLOAT\n";
  std::cout << "v1\n";
  test_operator<float, 1>(Q);
  std::cout << "v2\n";
  test_operator<float, 2>(Q);
  std::cout << "v3\n";
  test_operator<float, 3>(Q);
  std::cout << "v4\n";
  test_operator<float, 4>(Q);
  std::cout << "v8\n";
  test_operator<float, 8>(Q);
  std::cout << "v16\n";
  test_operator<float, 16>(Q);
  std::cout << std::endl;

  std::cout << "DOUBLE\n";
  std::cout << "v1\n";
  test_operator<double, 1>(Q);
  std::cout << "v2\n";
  test_operator<double, 2>(Q);
  std::cout << "v3\n";
  test_operator<double, 3>(Q);
  std::cout << "v4\n";
  test_operator<double, 4>(Q);
  std::cout << "v8\n";
  test_operator<double, 8>(Q);
  std::cout << "v16\n";
  test_operator<double, 16>(Q);
  std::cout << std::endl;





  return 0;
}
