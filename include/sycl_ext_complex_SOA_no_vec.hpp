// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Adapted from the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _SYCL_EXT_CPLX_COMPLEX
#define _SYCL_EXT_CPLX_COMPLEX

#define _SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD namespace sycl::ext::cplx {
#define _SYCL_EXT_CPLX_END_NAMESPACE_STD }
#define _SYCL_EXT_CPLX_INLINE_VISIBILITY                                       \
  inline __attribute__((__visibility__("hidden"), __always_inline__))

#include <complex>
#include <sstream> // for std::basic_ostringstream
#include <sycl/sycl.hpp>
#include <type_traits>

_SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD

template <class _v, class _Tp, int _N> class complex {
private:
  _v re;
  _v im;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void static complex_mul_(_Tp &__a, _Tp &__b,
                                                            _Tp &__c, _Tp &__d,
                                                            _Tp &__x,
                                                            _Tp &__y) {
    _Tp __ac = __a * __c;
    _Tp __bd = __b * __d;
    _Tp __ad = __a * __d;
    _Tp __bc = __b * __c;
    __x = __ac - __bd;
    __y = __ad + __bc;
    if (sycl::isnan(__x) && sycl::isnan(__y)) {
      bool __recalc = false;
      if (sycl::isinf(__a) || sycl::isinf(__b)) {
        __a = sycl::copysign(sycl::isinf(__a) ? _Tp(1) : _Tp(0), __a);
        __b = sycl::copysign(sycl::isinf(__b) ? _Tp(1) : _Tp(0), __b);
        if (sycl::isnan(__c))
          __c = sycl::copysign(_Tp(0), __c);
        if (sycl::isnan(__d))
          __d = sycl::copysign(_Tp(0), __d);
        __recalc = true;
      }
      if (sycl::isinf(__c) || sycl::isinf(__d)) {
        __c = sycl::copysign(sycl::isinf(__c) ? _Tp(1) : _Tp(0), __c);
        __d = sycl::copysign(sycl::isinf(__d) ? _Tp(1) : _Tp(0), __d);
        if (sycl::isnan(__a))
          __a = sycl::copysign(_Tp(0), __a);
        if (sycl::isnan(__b))
          __b = sycl::copysign(_Tp(0), __b);
        __recalc = true;
      }
      if (!__recalc && (sycl::isinf(__ac) || sycl::isinf(__bd) ||
                        sycl::isinf(__ad) || sycl::isinf(__bc))) {
        if (sycl::isnan(__a))
          __a = sycl::copysign(_Tp(0), __a);
        if (sycl::isnan(__b))
          __b = sycl::copysign(_Tp(0), __b);
        if (sycl::isnan(__c))
          __c = sycl::copysign(_Tp(0), __c);
        if (sycl::isnan(__d))
          __d = sycl::copysign(_Tp(0), __d);
        __recalc = true;
      }
      if (__recalc) {
        __x = _Tp(INFINITY) * (__a * __c - __b * __d);
        __y = _Tp(INFINITY) * (__a * __d + __b * __c);
      }
    }
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void static complex_div_(_Tp &__a, _Tp &__b,
                                                            _Tp &__c, _Tp &__d,
                                                            _Tp &__x,
                                                            _Tp &__y) {
    int __ilogbw = 0;
    _Tp __logbw = sycl::logb(sycl::fmax(sycl::fabs(__c), sycl::fabs(__d)));
    if (sycl::isfinite(__logbw)) {
      __ilogbw = static_cast<int>(__logbw);
      __c = sycl::ldexp(__c, -__ilogbw);
      __d = sycl::ldexp(__d, -__ilogbw);
    }
    _Tp __denom = __c * __c + __d * __d;
    __x = sycl::ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
    __y = sycl::ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (sycl::isnan(__x) && sycl::isnan(__y)) {
      if ((__denom == _Tp(0)) && (!sycl::isnan(__a) || !sycl::isnan(__b))) {
        __x = sycl::copysign(_Tp(INFINITY), __c) * __a;
        __y = sycl::copysign(_Tp(INFINITY), __c) * __b;
      } else if ((sycl::isinf(__a) || sycl::isinf(__b)) &&
                 sycl::isfinite(__c) && sycl::isfinite(__d)) {
        __a = sycl::copysign(sycl::isinf(__a) ? _Tp(1) : _Tp(0), __a);
        __b = sycl::copysign(sycl::isinf(__b) ? _Tp(1) : _Tp(0), __b);
        __x = _Tp(INFINITY) * (__a * __c + __b * __d);
        __y = _Tp(INFINITY) * (__b * __c - __a * __d);
      } else if (sycl::isinf(__logbw) && __logbw > _Tp(0) &&
                 sycl::isfinite(__a) && sycl::isfinite(__b)) {
        __c = sycl::copysign(sycl::isinf(__c) ? _Tp(1) : _Tp(0), __c);
        __d = sycl::copysign(sycl::isinf(__d) ? _Tp(1) : _Tp(0), __d);
        __x = _Tp(0) * (__a * __c + __b * __d);
        __y = _Tp(0) * (__b * __c - __a * __d);
      }
    }
  }

public:
  complex() : re(), im(){};
  complex(_v real, _v imag) : re(real), im(imag){};
  template <typename T>
  complex(std::complex<T> c) : re(c.real()), im(c.imag()){};

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _v real() const { return re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _v imag() const { return im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  void real(_v val) { re = val; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  void imag(_v val) { im = val; }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator+(const complex<_v, _Tp, _N> &__x, const complex<_v, _Tp, _N> &__y) {
    _v re;
    _v im;
#pragma unroll
    for (size_t i = 0; i < _N; ++i) {
      re[i] = __x.re[i] + __y.re[i];
      im[i] = __x.im[i] + __y.im[i];
    }

    return complex<_v, _Tp, _N>(re, im);
  }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator-(const complex<_v, _Tp, _N> &__x, const complex<_v, _Tp, _N> &__y) {
    _v re;
    _v im;
#pragma unroll
    for (size_t i = 0; i < _N; ++i) {
      re[i] = __x.re[i] - __y.re[i];
      im[i] = __x.im[i] - __y.im[i];
    }

    return complex<_v, _Tp, _N>(re, im);
  }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator*(const complex<_v, _Tp, _N> &__z, const complex<_v, _Tp, _N> &__w) {
    _v __a = __z.real();
    _v __b = __z.imag();
    _v __c = __w.real();
    _v __d = __w.imag();
    _v __x;
    _v __y;
#pragma unroll
    for (size_t i = 0; i < _N; ++i)
      complex_mul_(__a[i], __b[i], __c[i], __d[i], __x[i], __y[i]);

    return complex<_v, _Tp, _N>(__x, __y);
  }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator/(const complex<_v, _Tp, _N> &__z, const complex<_v, _Tp, _N> &__w) {
    _v __a = __z.real();
    _v __b = __z.imag();
    _v __c = __w.real();
    _v __d = __w.imag();
    _v __x;
    _v __y;
#pragma unroll
    for (size_t i = 0; i < _N; ++i)
      complex_div_(__a[i], __b[i], __c[i], __d[i], __x[i], __y[i]);

    return complex<_v, _Tp, _N>(__x, __y);
  }
};

// abs

template <class _v, class _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::vec<_Tp, _N>
abs(const complex<_v, _Tp, _N> &__c) {
  return sycl::hypot(__c.real(), __c.imag());
}

// arg

template <class _v, class _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::vec<_Tp, _N>
arg(const complex<_v, _Tp, _N> &__c) {
  return sycl::atan2(__c.imag(), __c.real());
}

// log

template <class _v, class _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
log(const complex<_v, _Tp, _N> &__x) {
  return complex<_v, _Tp, _N>(sycl::log(abs(__x)), arg(__x));
}

// exp

template <class _Tp> void exp_error_handling(_Tp &__e, _Tp &__re, _Tp &__im) {
  if (__im == 0) {
    __re = __e;
    __im = sycl::copysign(_Tp(0), __im);
    return;
  }
  if (sycl::isinf(__re)) {
    if (__re < _Tp(0)) {
      if (!sycl::isfinite(__im))
        __im = _Tp(1);
    } else if (__im == 0 || !sycl::isfinite(__im)) {
      if (sycl::isinf(__im))
        __im = _Tp(NAN);
      return;
    }
  }

  __re = __e * sycl::cos(__im);
  __im = __e * sycl::sin(__im);
  return;
}

template <class _v, class _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
exp(const complex<_v, _Tp, _N> &__x) {
  _v __e = sycl::exp(__x.real());
  _v __re = __x.real();
  _v __im = __x.imag();

#pragma unroll
  for (size_t i = 0; i < _N; ++i)
    exp_error_handling(__e[i], __re[i], __im[i]);

  return complex<_v, _Tp, _N>(__re, __im);
}

// pow

template <class _v, class _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
pow(const complex<_v, _Tp, _N> &__x, const complex<_v, _Tp, _N> &__y) {
  return exp(__y * log(__x));
}

_SYCL_EXT_CPLX_END_NAMESPACE_STD

#undef _SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_END_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_INLINE_VISIBILITY

#endif // _SYCL_EXT_CPLX_COMPLEX
