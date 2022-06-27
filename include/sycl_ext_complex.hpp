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

#ifdef AOS

using std::enable_if;
using std::integral_constant;
using std::is_floating_point;
using std::is_integral;
using std::is_same;

using std::basic_istream;
using std::basic_ostream;
using std::basic_ostringstream;

using std::declval;

template <bool _Val> using _BoolConstant = integral_constant<bool, _Val>;

template <class _Tp, class _Up>
using _IsNotSame = _BoolConstant<!__is_same(_Tp, _Up)>;

template <class _Tp> struct __numeric_type {
  static void __test(...);
  static sycl::half __test(sycl::half);
  static float __test(float);
  static double __test(char);
  static double __test(int);
  static double __test(unsigned);
  static double __test(long);
  static double __test(unsigned long);
  static double __test(long long);
  static double __test(unsigned long long);
  static double __test(double);

  typedef decltype(__test(declval<_Tp>())) type;
  static const bool value = _IsNotSame<type, void>::value;
};

template <> struct __numeric_type<void> { static const bool value = true; };

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&__numeric_type<_A2>::value
              &&__numeric_type<_A3>::value>
class __promote_imp {
public:
  static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
  typedef typename __promote_imp<_A3>::type __type3;

public:
  typedef decltype(__type1() + __type2() + __type3()) type;
  static const bool value = true;
};

template <class _A1, class _A2> class __promote_imp<_A1, _A2, void, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;

public:
  typedef decltype(__type1() + __type2()) type;
  static const bool value = true;
};

template <class _A1> class __promote_imp<_A1, void, void, true> {
public:
  typedef typename __numeric_type<_A1>::type type;
  static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

template <class _Tp> class complex;

template <class _Tp>
struct is_gencomplex
    : std::integral_constant<bool,
                             std::is_same_v<_Tp, complex<double>> ||
                                 std::is_same_v<_Tp, complex<float>> ||
                                 std::is_same_v<_Tp, complex<sycl::half>>> {};

template <class _Tp>
struct is_genfloat
    : std::integral_constant<bool, std::is_same_v<_Tp, double> ||
                                       std::is_same_v<_Tp, float> ||
                                       std::is_same_v<_Tp, sycl::half>> {};

template <class _Tp>
complex<_Tp> operator*(const complex<_Tp> &__z, const complex<_Tp> &__w);
template <class _Tp>
complex<_Tp> operator/(const complex<_Tp> &__x, const complex<_Tp> &__y);

template <class _Tp> class complex {
public:
  typedef _Tp value_type;

private:
  value_type __re_;
  value_type __im_;

public:
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      const value_type &__re = value_type(),
      const value_type &__im = value_type())
      : __re_(__re), __im_(__im) {}
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(const complex<_Xp> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      const std::complex<_Xp> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY operator std::complex<_Xp>() {
    return std::complex<_Xp>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr value_type imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const value_type &__re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(const value_type &__re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(const value_type &__re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(const value_type &__re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(const value_type &__re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <> class complex<float>;
template <> class complex<double>;

template <> class complex<sycl::half> {
  sycl::half __re_;
  sycl::half __im_;

public:
  typedef sycl::half value_type;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(
      sycl::half __re = sycl::half{}, sycl::half __im = sycl::half{})
      : __re_(__re), __im_(__im) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  explicit constexpr complex(const complex<float> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  explicit constexpr complex(const complex<double> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr operator std::complex<sycl::half>() {
    return std::complex<sycl::half>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr sycl::half real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr sycl::half imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(sycl::half __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(sycl::half __re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(sycl::half __re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(sycl::half __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(sycl::half __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <> class complex<float> {
  float __re_;
  float __im_;

public:
  typedef float value_type;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(float __re = 0.0f,
                                                     float __im = 0.0f)
      : __re_(__re), __im_(__im) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const complex<sycl::half> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  explicit constexpr complex(const complex<double> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const std::complex<float> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr operator std::complex<float>() {
    return std::complex<float>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr float real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr float imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(float __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(float __re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(float __re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(float __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(float __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

template <> class complex<double> {
  double __re_;
  double __im_;

public:
  typedef double value_type;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr complex(double __re = 0.0,
                                                     double __im = 0.0)
      : __re_(__re), __im_(__im) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const complex<sycl::half> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const complex<float> &__c);
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr complex(const std::complex<double> &__c)
      : __re_(__c.real()), __im_(__c.imag()) {}
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  constexpr operator std::complex<double>() {
    return std::complex<double>(__re_, __im_);
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr double real() const {
    return __re_;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr double imag() const {
    return __im_;
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void real(value_type __re) { __re_ = __re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY void imag(value_type __im) { __im_ = __im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(double __re) {
    __re_ = __re;
    __im_ = value_type();
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator+=(double __re) {
    __re_ += __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator-=(double __re) {
    __re_ -= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator*=(double __re) {
    __re_ *= __re;
    __im_ *= __re;
    return *this;
  }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator/=(double __re) {
    __re_ /= __re;
    __im_ /= __re;
    return *this;
  }

  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &operator=(const complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator=(const std::complex<_Xp> &__c) {
    __re_ = __c.real();
    __im_ = __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator+=(const complex<_Xp> &__c) {
    __re_ += __c.real();
    __im_ += __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator-=(const complex<_Xp> &__c) {
    __re_ -= __c.real();
    __im_ -= __c.imag();
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator*=(const complex<_Xp> &__c) {
    *this = *this * complex(__c.real(), __c.imag());
    return *this;
  }
  template <class _Xp>
  _SYCL_EXT_CPLX_INLINE_VISIBILITY complex &
  operator/=(const complex<_Xp> &__c) {
    *this = *this / complex(__c.real(), __c.imag());
    return *this;
  }
};

inline constexpr complex<sycl::half>::complex(const complex<float> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<sycl::half>::complex(const complex<double> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<float>::complex(const complex<sycl::half> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<float>::complex(const complex<double> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<double>::complex(const complex<sycl::half> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline constexpr complex<double>::complex(const complex<float> &__c)
    : __re_(__c.real()), __im_(__c.imag()) {}

// 26.3.6 operators:

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator+(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator+(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  complex<_Tp> __t(__x);
  __t += __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator+(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__y);
  __t += __x;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator-(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator-(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  complex<_Tp> __t(__x);
  __t -= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator-(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(-__y);
  __t += __x;
  return __t;
}

template <class _Tp>
complex<_Tp> operator*(const complex<_Tp> &__z, const complex<_Tp> &__w) {
  _Tp __a = __z.real();
  _Tp __b = __z.imag();
  _Tp __c = __w.real();
  _Tp __d = __w.imag();
  _Tp __ac = __a * __c;
  _Tp __bd = __b * __d;
  _Tp __ad = __a * __d;
  _Tp __bc = __b * __c;
  _Tp __x = __ac - __bd;
  _Tp __y = __ad + __bc;
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
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator*(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  complex<_Tp> __t(__x);
  __t *= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator*(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__y);
  __t *= __x;
  return __t;
}

template <class _Tp>
complex<_Tp> operator/(const complex<_Tp> &__z, const complex<_Tp> &__w) {
  int __ilogbw = 0;
  _Tp __a = __z.real();
  _Tp __b = __z.imag();
  _Tp __c = __w.real();
  _Tp __d = __w.imag();
  _Tp __logbw = sycl::logb(sycl::fmax(sycl::fabs(__c), sycl::fabs(__d)));
  if (sycl::isfinite(__logbw)) {
    __ilogbw = static_cast<int>(__logbw);
    __c = sycl::ldexp(__c, -__ilogbw);
    __d = sycl::ldexp(__d, -__ilogbw);
  }
  _Tp __denom = __c * __c + __d * __d;
  _Tp __x = sycl::ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
  _Tp __y = sycl::ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (sycl::isnan(__x) && sycl::isnan(__y)) {
    if ((__denom == _Tp(0)) && (!sycl::isnan(__a) || !sycl::isnan(__b))) {
      __x = sycl::copysign(_Tp(INFINITY), __c) * __a;
      __y = sycl::copysign(_Tp(INFINITY), __c) * __b;
    } else if ((sycl::isinf(__a) || sycl::isinf(__b)) && sycl::isfinite(__c) &&
               sycl::isfinite(__d)) {
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
  return complex<_Tp>(__x, __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp> operator/(const complex<_Tp> &__x,
                                                        const _Tp &__y) {
  return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator/(const _Tp &__x, const complex<_Tp> &__y) {
  complex<_Tp> __t(__x);
  __t /= __y;
  return __t;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator+(const complex<_Tp> &__x) {
  return __x;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
operator-(const complex<_Tp> &__x) {
  return complex<_Tp>(-__x.real(), -__x.imag());
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator==(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator==(const complex<_Tp> &__x, const _Tp &__y) {
  return __x.real() == __y && __x.imag() == 0;
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator==(const _Tp &__x, const complex<_Tp> &__y) {
  return __x == __y.real() && 0 == __y.imag();
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator!=(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  return !(__x == __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator!=(const complex<_Tp> &__x, const _Tp &__y) {
  return !(__x == __y);
}

template <class _Tp>
_SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr bool
operator!=(const _Tp &__x, const complex<_Tp> &__y) {
  return !(__x == __y);
}

// 26.3.7 values:

template <class _Tp, bool = is_integral<_Tp>::value,
          bool = is_floating_point<_Tp>::value>
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp> struct __libcpp_complex_overload_traits<_Tp, true, false> {
  typedef double _ValueType;
  typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp> struct __libcpp_complex_overload_traits<_Tp, false, true> {
  typedef _Tp _ValueType;
  typedef complex<_Tp> _ComplexType;
};

// real

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _Tp
real(const complex<_Tp> &__c) {
  return __c.real();
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename __libcpp_complex_overload_traits<_Tp>::_ValueType
    real(_Tp __re) {
  return __re;
}

// imag

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _Tp
imag(const complex<_Tp> &__c) {
  return __c.imag();
}

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr
    typename __libcpp_complex_overload_traits<_Tp>::_ValueType
    imag(_Tp) {
  return 0;
}

// Vec complex
template <typename _Tp, int _N> class vec {
private:
  _Tp _Data[_N];

public:
  vec(){};
  vec(_Tp val) {

#pragma unroll
    for (int i = 0; i < _N; ++i)
      _Data[i] = val;
  }

  static constexpr size_t size() { return _N; }
  _Tp &operator[](int index) { return _Data[index]; }
  const _Tp &operator[](int index) const { return _Data[index]; }

  friend vec operator+(const vec &lhs, const vec &rhs) {
    vec<_Tp, _N> out;

#pragma unroll
    for (int i = 0; i < _N; ++i)
      out[i] = lhs[i] + rhs[i];

    return out;
  }

  friend vec operator-(const vec &lhs, const vec &rhs) {
    vec<_Tp, _N> out;

#pragma unroll
    for (int i = 0; i < _N; ++i)
      out[i] = lhs[i] - rhs[i];

    return out;
  }

  friend vec operator*(const vec &lhs, const vec &rhs) {
    vec<_Tp, _N> out;

#pragma unroll
    for (int i = 0; i < _N; ++i)
      out[i] = lhs[i] * rhs[i];

    return out;
  }

  friend vec operator/(const vec &lhs, const vec &rhs) {
    vec<_Tp, _N> out;

#pragma unroll
    for (int i = 0; i < _N; ++i)
      out[i] = lhs[i] / rhs[i];

    return out;
  }
};

// abs

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp
abs(const complex<_Tp> &__c) {
  return sycl::hypot(__c.real(), __c.imag());
}

template <typename _Tp, typename _cTp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::vec<_cTp, _N>
abs(const vec<_Tp, _N> &__c) {
  sycl::vec<_cTp, _N> out;

#pragma unroll
  for (int i = 0; i < _N; ++i)
    out[i] = abs(__c[i]);

  return out;
}

// arg

template <class _Tp>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY _Tp
arg(const complex<_Tp> &__c) {
  return sycl::atan2(__c.imag(), __c.real());
}

template <typename _Tp, typename _cTp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY sycl::vec<_cTp, _N>
arg(const vec<_Tp, _N> &__c) {
  sycl::vec<_cTp, _N> out;

#pragma unroll
  for (int i = 0; i < _N; ++i)
    out[i] = arg(__c[i]);

  return out;
}

// log

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
log(const complex<_Tp> &__x) {
  return complex<_Tp>(sycl::log(abs(__x)), arg(__x));
}

template <typename _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY vec<_Tp, _N>
log(const vec<_Tp, _N> &__c) {
  vec<_Tp, _N> out;

#pragma unroll
  for (int i = 0; i < _N; ++i)
    out[i] = log(__c[i]);

  return out;
}

// exp

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL complex<_Tp> exp(const complex<_Tp> &__x) {
  _Tp __i = __x.imag();
  if (__i == 0) {
    return complex<_Tp>(sycl::exp(__x.real()),
                        sycl::copysign(_Tp(0), __x.imag()));
  }
  if (sycl::isinf(__x.real())) {
    if (__x.real() < _Tp(0)) {
      if (!sycl::isfinite(__i))
        __i = _Tp(1);
    } else if (__i == 0 || !sycl::isfinite(__i)) {
      if (sycl::isinf(__i))
        __i = _Tp(NAN);
      return complex<_Tp>(__x.real(), __i);
    }
  }
  _Tp __e = sycl::exp(__x.real());
  return complex<_Tp>(__e * sycl::cos(__i), __e * sycl::sin(__i));
}

template <typename _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY vec<_Tp, _N>
exp(const vec<_Tp, _N> &__c) {
  vec<_Tp, _N> out;

#pragma unroll
  for (int i = 0; i < _N; ++i)
    out[i] = exp(__c[i]);

  return out;
}

// pow

template <class _Tp, class = std::enable_if<is_gencomplex<_Tp>::value>>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_Tp>
pow(const complex<_Tp> &__x, const complex<_Tp> &__y) {
  return exp(__y * log(__x));
}

template <typename _Tp, int _N>
SYCL_EXTERNAL _SYCL_EXT_CPLX_INLINE_VISIBILITY vec<_Tp, _N>
pow(const vec<_Tp, _N> &__x, const vec<_Tp, _N> &__y) {
  vec<_Tp, _N> out;

#pragma unroll
  for (int i = 0; i < _N; ++i)
    out[i] = pow(__x[i], __y[i]);

  return out;
}

// operator <<

template <class _Tp, class _CharT, class _Traits>
basic_ostream<_CharT, _Traits> &operator<<(basic_ostream<_CharT, _Traits> &__os,
                                           const complex<_Tp> &__x) {
  basic_ostringstream<_CharT, _Traits> __s;
  __s.flags(__os.flags());
  __s.imbue(__os.getloc());
  __s.precision(__os.precision());
  __s << '(' << __x.real() << ',' << __x.imag() << ')';
  return __os << __s.str();
}

#else // AOS

template <class _v, class _Tp, int _N>
class complex {
private:
  _v re;
  _v im;

  _SYCL_EXT_CPLX_INLINE_VISIBILITY void static mul_recalc(_Tp &__a, _Tp &__b,
                                                          _Tp &__c, _Tp &__d,
                                                          _Tp &__ac, _Tp &__bd,
                                                          _Tp &__ad, _Tp &__bc,
                                                          _Tp &__x, _Tp &__y) {
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

  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  void static div_isfinite(int &__ilogbw, _Tp &__logbw, _Tp &__c, _Tp &__d) {
    if (sycl::isfinite(__logbw)) {
      __ilogbw = static_cast<int>(__logbw);
      __c = sycl::ldexp(__c, -__ilogbw);
      __d = sycl::ldexp(__d, -__ilogbw);
    }
  }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  void static div_isnan(_Tp &__denom, _Tp &__logbw, _Tp &__a, _Tp &__b,
                        _Tp &__c, _Tp &__d, _Tp &__x, _Tp &__y) {
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
  template <typename T> complex(std::complex<T> c) : re(c.real()), im(c.imag()){};

  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _v real() const { return re; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY constexpr _v imag() const { return im; }

  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  void real(_v val) { re = val; }
  _SYCL_EXT_CPLX_INLINE_VISIBILITY
  void imag(_v val) { im = val; }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator+(const complex<_v, _Tp, _N> &__x, const complex<_v, _Tp, _N> &__y) {
    return complex<_v, _Tp, _N>(__x.re + __y.re, __x.im + __y.im);
  }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator-(const complex<_v, _Tp, _N> &__x, const complex<_v, _Tp, _N> &__y) {
    return complex<_v, _Tp, _N>(__x.re - __y.re, __x.im - __y.im);
  }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator*(const complex<_v, _Tp, _N> &__z, const complex<_v, _Tp, _N> &__w) {
    // vectorisable
    _v __a = __z.real();
    _v __b = __z.imag();
    _v __c = __w.real();
    _v __d = __w.imag();
    _v __ac = __a * __c;
    _v __bd = __b * __d;
    _v __ad = __a * __d;
    _v __bc = __b * __c;
    _v __x = __ac - __bd;
    _v __y = __ad + __bc;

// not vectorisable
#pragma unroll
    for (size_t i = 0; i < _N; ++i)
      mul_recalc(__a[i], __b[i], __c[i], __d[i], __ac[i], __bd[i], __ad[i],
                 __bc[i], __x[i], __y[i]);

    return complex<_v, _Tp, _N>(__x, __y);
  }

  friend _SYCL_EXT_CPLX_INLINE_VISIBILITY complex<_v, _Tp, _N>
  operator/(const complex<_v, _Tp, _N> &__z, const complex<_v, _Tp, _N> &__w) {
    // vectorisable
    sycl::vec<int, _N> __ilogbw(0);
    _v __a = __z.real();
    _v __b = __z.imag();
    _v __c = __w.real();
    _v __d = __w.imag();

    // I don't think this is standard sycl, and may rely on unique DPCPP stuff
    // For a generic solution this may not be usable
    _v __logbw = sycl::logb(sycl::fmax(sycl::fabs(__c), sycl::fabs(__d)));

// not vectorisable
#pragma unroll
    for (size_t i = 0; i < _N; ++i)
      div_isfinite(__ilogbw[i], __logbw[i], __c[i], __d[i]);

    // vectorisable
    _v __denom = __c * __c + __d * __d;
    _v __x = sycl::ldexp((__a * __c + __b * __d) / __denom, -__ilogbw);
    _v __y = sycl::ldexp((__b * __c - __a * __d) / __denom, -__ilogbw);

// not vectorisable
#pragma unroll
    for (size_t i = 0; i < _N; ++i)
      div_isnan(__denom[i], __logbw[i], __a[i], __b[i], __c[i], __d[i], __x[i],
                __y[i]);

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

#endif // AOS

_SYCL_EXT_CPLX_END_NAMESPACE_STD

#undef _SYCL_EXT_CPLX_BEGIN_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_END_NAMESPACE_STD
#undef _SYCL_EXT_CPLX_INLINE_VISIBILITY

#endif // _SYCL_EXT_CPLX_COMPLEX
