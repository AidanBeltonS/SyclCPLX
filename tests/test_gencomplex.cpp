#include "sycl_ext_complex.hpp"
#include <sycl/sycl.hpp>

using namespace sycl::ext::cplx;

// Check is_gencomplex
void check_is_gencomplex() {
  static_assert(is_gencomplex<complex<double>>::value == true);
  static_assert(is_gencomplex<complex<float>>::value == true);
  static_assert(is_gencomplex<complex<sycl::half>>::value == true);

  static_assert(is_gencomplex<complex<long long>>::value == false);
  static_assert(is_gencomplex<complex<long>>::value == false);
  static_assert(is_gencomplex<complex<int>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned int>>::value == false);

  static_assert(is_gencomplex_v<complex<double>> == true);
  static_assert(is_gencomplex_v<complex<float>> == true);
  static_assert(is_gencomplex_v<complex<sycl::half>> == true);

  static_assert(is_gencomplex_v<complex<long long>> == false);
  static_assert(is_gencomplex_v<complex<long>> == false);
  static_assert(is_gencomplex_v<complex<int>> == false);
  static_assert(is_gencomplex_v<complex<unsigned long long>> == false);
  static_assert(is_gencomplex_v<complex<unsigned long>> == false);
  static_assert(is_gencomplex_v<complex<unsigned int>> == false);
}

int main() {
  check_is_gencomplex();
  return 0;
}
