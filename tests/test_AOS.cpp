#include <cmath>
#include <iomanip>
#include <sycl/sycl.hpp>

#include "sycl_ext_complex_AOS.hpp"

int main() {
    sycl::ext::cplx::vec<sycl::ext::cplx::complex<float>, 4> A(sycl::ext::cplx::complex<float>(3, 4));
    sycl::ext::cplx::vec<sycl::ext::cplx::complex<float>, 4> B(sycl::ext::cplx::complex<float>(3, 4));
    sycl::ext::cplx::vec<sycl::ext::cplx::complex<float>, 4> C;
    sycl::vec<float, 4> D;

    C = A + B;
    std::cout << C[0] << std::endl;

    C = A - B;
    std::cout << C[0] << std::endl;

    C = A * B;
    std::cout << C[0] << std::endl;

    C = A / B;
    std::cout << C[0] << std::endl;

    D = sycl::ext::cplx::abs<sycl::ext::cplx::complex<float>, float, 4>(A);
    std::cout << D[0] << std::endl;

    D = sycl::ext::cplx::arg<sycl::ext::cplx::complex<float>, float, 4>(A);
    std::cout << D[0] << std::endl;

    C = sycl::ext::cplx::log(A);
    std::cout << C[0] << std::endl;

    C = sycl::ext::cplx::exp(A);
    std::cout << C[0] << std::endl;

    C = sycl::ext::cplx::pow(A, B);
    std::cout << C[0] << std::endl;

    return 0;
}
