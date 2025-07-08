/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2023, ITU/ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
 *    be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once
#include <cmath>
#include "layer.h"
#if __AVX2__
#include <immintrin.h>
#endif

namespace sadl
{
namespace layers
{
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2x2
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T> template<int s_h, int s_w> void Conv2D<T>::conv2d_2x2_s_dispatch(const Tensor<T> &A, const Tensor<T> &kernel)
{
#if __AVX2__
#define CONV_MOD8 simd8_conv2d_2x2_s_d
#define CONV_MOD16 simd16_conv2d_2x2_s_d
#define CONV_MOD32 simd32_conv2d_2x2_s_d
#else
#define CONV_MOD8 conv2d_2x2_s_d
#define CONV_MOD16 conv2d_2x2_s_d
#define CONV_MOD32 conv2d_2x2_s_d
#endif

  const int in_D     = A.dims()[3];
  const int cin_by_g = in_D / m_groups;
  switch (cin_by_g)
  {
  case 1:
    conv2d_2x2_s_d<1, s_h, s_w>(A, kernel);
    break;
  case 2:
    conv2d_2x2_s_d<2, s_h, s_w>(A, kernel);
    break;
  case 4:
    conv2d_2x2_s_d<4, s_h, s_w>(A, kernel);
    break;
  case 6:
    conv2d_2x2_s_d<6, s_h, s_w>(A, kernel);
    break;
  case 34:
    conv2d_2x2_s_d<34, s_h, s_w>(A, kernel);
    break;
  case 8:
    CONV_MOD8<8, s_h, s_w>(A, kernel);
    break;
  case 16:
    CONV_MOD16<16, s_h, s_w>(A, kernel);
    break;
  case 24:
    CONV_MOD8<24, s_h, s_w>(A, kernel);
    break;
  case 32:
    CONV_MOD32<32, s_h, s_w>(A, kernel);
    break;
  case 48:
    CONV_MOD16<48, s_h, s_w>(A, kernel);
    break;
  case 64:
    CONV_MOD32<64, s_h, s_w>(A, kernel);
    break;
  case 72:
    // better do 64 and than 8
    CONV_MOD8<72, s_h, s_w>(A, kernel);
    break;
  case 96:
    CONV_MOD32<96, s_h, s_w>(A, kernel);
    break;
  case 112:
    CONV_MOD16<112, s_h, s_w>(A, kernel);
    break;
  case 128:
    CONV_MOD32<128, s_h, s_w>(A, kernel);
    break;
  case 144:
    CONV_MOD16<144, s_h, s_w>(A, kernel);
    break;
  case 160:
    CONV_MOD32<160, s_h, s_w>(A, kernel);
    break;
  case 176:
    CONV_MOD16<176, s_h, s_w>(A, kernel);
    break;
  case 192:
    CONV_MOD32<192, s_h, s_w>(A, kernel);
    break;
  case 272:
    CONV_MOD16<272, s_h, s_w>(A, kernel);
    break;
  case 288:
    CONV_MOD32<288, s_h, s_w>(A, kernel);
    break;
  case 384:
    CONV_MOD32<384, s_h, s_w>(A, kernel);
    break;
  case 480:
    CONV_MOD32<480, s_h, s_w>(A, kernel);
    break;
  default:
    conv2d_2x2_s<s_h, s_w>(A, kernel);
    break;
  }
#undef CONV_MOD8
#undef CONV_MOD16
#undef CONV_MOD32
}

template<typename T> template<int s_h, int s_w> void Conv2D<T>::conv2d_2x2_s(const Tensor<T> &A, const Tensor<T> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int k_size_i{ 2 };
  constexpr int k_size_j{ 2 };
  //assert(half_size_i == m_pads[0]);
  //assert(half_size_j == m_pads[1]);
  constexpr int start_h{ 0 };
  constexpr int start_w{ 0 };
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv2d_2x2_s inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W
            << " groups=" << m_groups << " " << (in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w)) / (1000 * m_groups)
            << " kMAC" << std::endl;
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif
  constexpr int im_nb     = 0;
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = 0; filter_i < k_size_i; ++filter_i)
        {
          // fixed
          for (int filter_j = 0; filter_j < k_size_j; ++filter_j)
          {
            // fixed
            for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = filter_i;
              int kj = filter_j;
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
              } else {
                COUNTERS_MAC_NOP(1);
              }
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

template<typename T> template<int in_D, int s_h, int s_w> void Conv2D<T>::conv2d_2x2_s_d(const Tensor<T> &A, const Tensor<T> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int k_size_i{ 2 };
  constexpr int k_size_j{ 2 };
  // assert(half_size_i == m_pads[0]);
  // assert(half_size_j == m_pads[1]);
  constexpr int start_h{ 0 };
  constexpr int start_w{ 0 };

#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv2d_2x2_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W
            << " groups=" << m_groups << " " << (in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w)) / (1000 * m_groups)
            << " kMAC" << std::endl;
#endif
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif

  constexpr int im_nb     = 0;
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = 0; filter_i < k_size_i; ++filter_i)
        {
          // fixed
          for (int filter_j = 0; filter_j < k_size_j; ++filter_j)
          {
            // fixed
            for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = filter_i;
              int kj = filter_j;
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
              }
              else
              {
                COUNTERS_MAC_NOP(1);
              }
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

#if __AVX2__
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// 2x2
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
template<> template<int in_D, int s_h, int s_w> inline void Conv2D<float>::simd8_conv2d_2x2_s_d(const Tensor<float> &A, const Tensor<float> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int k_size_i{ 2 };
  constexpr int k_size_j{ 2 };
  // assert(half_size_i == m_pads[0]);
  // assert(half_size_j == m_pads[1]);
  constexpr int start_h{ 0 };
  constexpr int start_w{ 0 };
  constexpr int im_nb     = 0;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  assert((cin_by_g % 8 == 0) && "in_D / m_groups should be aligned to 8.");
  
#if DEBUG_SIMD && __AVX512F__
  if (cin_by_g >= 16)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv2d_2x2_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x'
              << in_W << " groups=" << m_groups << " "
              << (in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w)) / (1000 * m_groups) << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif

  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        __m256 sum = _mm256_setzero_ps();

        for (int filter_i = 0; filter_i < k_size_i; ++filter_i)
        {
          for (int filter_j = 0; filter_j < k_size_j; ++filter_j)
          {
            int ii = im_i + filter_i;
            int jj = im_j + filter_j;
            int ki = filter_i;
            int kj = filter_j;

            const float *aptr = A.addr(im_nb, ii, jj, offset);
            const float *kptr = kernel.addr(ki, kj, filter, 0);

            for (int filter_d = 0; filter_d < cin_by_g; filter_d += 8)
            {
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                __m256 a = _mm256_load_ps(aptr + filter_d);    // Aligned load
                __m256 k = _mm256_loadu_ps(kptr + filter_d);   // Not always aligned
#if __FMA__
                sum = _mm256_fmadd_ps(a, k, sum);
#else
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, k));
#endif
              }
            }
          }
        }
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = sum8_float(sum);
      }
    }
  }
}

#if __AVX512F__
template<> template<int in_D, int s_h, int s_w> inline void Conv2D<float>::simd16_conv2d_2x2_s_d(const Tensor<float> &A, const Tensor<float> &kernel)
{
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int k_size_i{ 2 };
  constexpr int k_size_j{ 2 };
  // assert(half_size_i == m_pads[0]);
  // assert(half_size_j == m_pads[1]);
  constexpr int start_h{ 0 };
  constexpr int start_w{ 0 };
  constexpr int im_nb     = 0;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  assert((cin_by_g % 16 == 0) && "in_D / m_groups should be aligned to 16 for AVX512.");
  
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif

  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        __m512 sum = _mm512_setzero_ps();

        for (int filter_i = 0; filter_i < k_size_i; ++filter_i)
        {
          for (int filter_j = 0; filter_j < k_size_j; ++filter_j)
          {
            int ii = im_i + filter_i;
            int jj = im_j + filter_j;
            int ki = filter_i;
            int kj = filter_j;

            const float *aptr = A.addr(im_nb, ii, jj, offset);
            const float *kptr = kernel.addr(ki, kj, filter, 0);

            for (int filter_d = 0; filter_d < cin_by_g; filter_d += 16)
            {
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                __m512 a = _mm512_load_ps(aptr + filter_d);    // Aligned load
                __m512 k = _mm512_loadu_ps(kptr + filter_d);   // Not always aligned
#if __FMA__
                sum = _mm512_fmadd_ps(a, k, sum);
#else
                sum = _mm512_add_ps(sum, _mm512_mul_ps(a, k));
#endif
              }
            }
          }
        }
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = sum16_float(sum);
      }
    }
  }
}
#endif

// int16
template<> template<int in_D, int s_h, int s_w> void Conv2D<int16_t>::simd8_conv2d_2x2_s_d(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{   // should be sse42
  using T = int16_t;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int k_size_i{ 2 };
  constexpr int k_size_j{ 2 };
  // assert(half_size_i == m_pads[0]);
  // assert(half_size_j == m_pads[1]);
  constexpr int start_h{ 0 };
  constexpr int start_w{ 0 };
  constexpr int im_nb     = 0;
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  assert((cin_by_g % 8 == 0) && "in_D / m_groups should be aligned to 8.");
  
#if DEBUG_SIMD && __AVX2__
    if (cin_by_g >= 8)
  {
    std::cout << "\n[WARN] suboptimal SIMD8 version conv2d_2x2_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x'
              << in_W << " groups=" << m_groups << " "
              << (in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w)) / (1000 * m_groups) << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif

  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        __m128i sum = _mm_setzero_si128();

        for (int filter_i = 0; filter_i < k_size_i; ++filter_i)
        {
          for (int filter_j = 0; filter_j < k_size_j; ++filter_j)
          {
            int ii = im_i + filter_i;
            int jj = im_j + filter_j;
            int ki = filter_i;
            int kj = filter_j;

            const int16_t *aptr = A.addr(im_nb, ii, jj, offset);
            const int16_t *kptr = kernel.addr(ki, kj, filter, 0);

            for (int filter_d = 0; filter_d < cin_by_g; filter_d += 8)
            {
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                __m128i a = _mm_load_si128((const __m128i *)(aptr + filter_d));    // Aligned load
                __m128i k = _mm_loadu_si128((const __m128i *)(kptr + filter_d));   // Not always aligned
                sum = _mm_add_epi32(sum, _mm_madd_epi16(a, k));
              }
            }
          }
        }
        typename ComputationType<T>::type z = (sum32_int16(sum) >> shift);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }
}

template<> template<int in_D, int s_h, int s_w> void Conv2D<int16_t>::simd16_conv2d_2x2_s_d(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  using T = int16_t;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int k_size_i{ 2 };
  constexpr int k_size_j{ 2 };
  constexpr int start_h{ 0 };
  constexpr int start_w{ 0 };
  constexpr int im_nb     = 0;
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  assert((cin_by_g % 16 == 0) && "in_D / m_groups should be aligned to 16.");
  
#if DEBUG_SIMD && __AVX512BW__
  if (cin_by_g >= 16)
  {
    std::cout << "\n[INFO] suboptimal SIMD16 version simd16_conv2d_2x2_s_d inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] "
              << in_H << 'x' << in_W << " groups=" << m_groups << " "
              << (in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w)) / (1000 * m_groups) << " kMAC" << std::endl;
  }
#endif
#if DEBUG_PATH
  std::cout<<__PRETTY_FUNCTION__<<std::endl;
#endif

  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        __m256i sum = _mm256_setzero_si256();

        for (int filter_i = 0; filter_i < k_size_i; ++filter_i)
        {
          for (int filter_j = 0; filter_j < k_size_j; ++filter_j)
          {
            int ii = im_i + filter_i;
            int jj = im_j + filter_j;
            int ki = filter_i;
            int kj = filter_j;

            const int16_t *aptr = A.addr(im_nb, ii, jj, offset);
            const int16_t *kptr = kernel.addr(ki, kj, filter, 0);

            for (int filter_d = 0; filter_d < cin_by_g; filter_d += 16)
            {
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                __m256i a = _mm256_load_si256((const __m256i *)(aptr + filter_d));    // Aligned load
                __m256i k = _mm256_loadu_si256((const __m256i *)(kptr + filter_d));   // Not always aligned
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(a, k));
              }
            }
          }
        }
        typename ComputationType<T>::type z = (sum32_int16(sum) >> shift);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<int16_t>(z);
      }
    }
  }
}

#if __AVX512BW__
template<> template<int in_D, int s_h, int s_w> void Conv2D<int16_t>::simd32_conv2d_2x2_s_d(const Tensor<int16_t> &A, const Tensor<int16_t> &kernel)
{
  using T = int16_t;
  const int     in_H{ A.dims()[1] };
  const int     in_W{ A.dims()[2] };
  const int     nb_filters{ kernel.dims()[2] };
  constexpr int k_size_i{ 2 };
  constexpr int k_size_j{ 2 };
  constexpr int start_h{ 0 };
  constexpr int start_w{ 0 };
  constexpr int im_nb     = 0;
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  assert((cin_by_g % 32 == 0) && "in_D / m_groups must be divisible by 32 for AVX512.");
  
#if DEBUG_PATH
  std::cout << __PRETTY_FUNCTION__ << std::endl;
#endif

  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;

    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        __m512i sum = _mm512_setzero_si512();

        for (int filter_i = 0; filter_i < k_size_i; ++filter_i)
        {
          for (int filter_j = 0; filter_j < k_size_j; ++filter_j)
          {
            int ii = im_i + filter_i;
            int jj = im_j + filter_j;
            int ki = filter_i;
            int kj = filter_j;

            const int16_t *aptr = A.addr(im_nb, ii, jj, offset);
            const int16_t *kptr = kernel.addr(ki, kj, filter, 0);

            for (int filter_d = 0; filter_d < cin_by_g; filter_d += 32)
            {
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                __m512i a = _mm512_load_si512((const __m512i *) (aptr + filter_d));    // Aligned load
                __m512i k = _mm512_loadu_si512((const __m512i *) (kptr + filter_d));   // Not always aligned
                sum       = _mm512_add_epi32(sum, _mm512_madd_epi16(a, k));
              }
            }
          }
        }
        typename ComputationType<int32_t>::type z = (_mm512_reduce_add_epi32(sum) >> shift);
        COUNTERS(z);
        SATURATE(z);
        m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(z);
      }
    }
  }
}
#endif
#endif
}   // namespace layers
}   // namespace sadl
