#ifndef CONFIG_H
#define CONFIG_H

#include <ccglib/fp16.h>
#include <ccglib/precision.h>

enum Size { small, large };


template<ccglib::ValueType Tin, ccglib::ValueType Tout, Size size> struct Params {
};

template <> struct Params<ccglib::float16, ccglib::float32, large> {
  static constexpr size_t B = 1;
  static constexpr size_t M = 8192;
  static constexpr size_t N = 8192;
  static constexpr size_t K = 8192;
};

#endif  // CONFIG_H
