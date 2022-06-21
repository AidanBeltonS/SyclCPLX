#include <cmath>
#include <iomanip>
#include <complex>

#include <sycl/sycl.hpp>

#include "sycl_ext_complex.hpp"


int main() {
  sycl::vec<float, 4> re(3);
  sycl::vec<float, 4> im(4);


  sycl::ext::cplx::complex<sycl::vec<float, 4>, float, 4> A(re, im);
  sycl::ext::cplx::complex<sycl::vec<float, 4>, float, 4> B(re, im);
  sycl::ext::cplx::complex<sycl::vec<float, 4>, float, 4> C;

  // Operator +
  C = A + B;
  std::cout << C.real()[0] << "," << C.imag()[0] << std::endl;

  // Operator -
  C = A - B;
  std::cout << C.real()[0] << "," << C.imag()[0] << std::endl;

  // Operator *
  C = A * B;
  std::cout << C.real()[0] << "," << C.imag()[0] << std::endl;

  // Operator /
  C = A / B;
  std::cout << C.real()[0] << "," << C.imag()[0] << std::endl;

  // abs
  sycl::vec<float, 4> D = sycl::ext::cplx::abs(A);
  std::cout << D[0] << std::endl;

  // arg
  D = sycl::ext::cplx::arg(A);
  std::cout << D[0] << std::endl;

  // log
  C = sycl::ext::cplx::log(A);
  std::cout << C.real()[0] << "," << C.imag()[0] << std::endl;

  // exp
  C = sycl::ext::cplx::exp(A);
  std::cout << C.real()[0] << "," << C.imag()[0] << std::endl;

  // pow
  C = sycl::ext::cplx::pow(A, B);
  std::cout << C.real()[0] << "," << C.imag()[0] << std::endl;

  return 0;
}