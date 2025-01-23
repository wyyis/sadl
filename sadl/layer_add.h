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
#include "layer.h"

namespace sadl
{
namespace layers
{
template<typename T> class Add : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
};

// TODO : check all dims, check loop for optiz
template<typename T> bool Add<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  if (in[0] == in[1])
  {
    std::cerr << "  input aliasing" << std::endl;
    return false;
  }

  int shift0, shift1, qfinal;
  if (in[0]->quantizer < in[1]->quantizer)
  {
    shift0 = 0;
    shift1 = in[1]->quantizer - in[0]->quantizer;
    qfinal = in[0]->quantizer;
  }
  else
  {
    shift0 = in[0]->quantizer - in[1]->quantizer;
    shift1 = 0;
    qfinal = in[1]->quantizer;
  }
  swap(*in[0], m_out);
  // adapt output width to second input (which are the bias) in order to be able to rescale as desired the input
  m_out.quantizer = qfinal;

  if (shift0 > 0)
  {
    const int shift = shift0;
    if (in[0]->dims() == in[1]->dims())
    {
#if DEBUG_MODEL_ANALYZE
      std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
      for (auto it0 = m_out.begin(), it1 = in[1]->begin(); it0 != m_out.end(); ++it0, ++it1)
      {
        typename ComputationType<T>::type z = *it0;
        ComputationType<T>::quantize(z, shift);
        z += *it1;
        COUNTERS(z);
        SATURATE(z);
        *it0 = static_cast<T>(z);
      }
    }
    else
    {
      if (in[1]->size() == 1)
      {   // ie in[0]->dims().size() == 1? happen if in[1] is a Const
        const Tensor<T> &B     = *in[1];
#if __AVX2__
        if constexpr (std::is_same_v<T, int16_t>)
        {
          if (m_out.size() % 16 == 0) 
          {
            const __m256i     value = _mm256_set1_epi16(B[0]);
            const __m256i     max   = _mm256_set1_epi16(32767);
            const __m256i     min   = _mm256_set1_epi16(-32768);
            T *a_ptr = m_out.data();
            for (int64_t i = 0; i < m_out.size(); i += 16, a_ptr+=16)
            {
              __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_ptr));
              __m256i result = _mm256_adds_epi16(x, value);
              result = _mm256_srai_epi16(result, shift);
#if SATURATE_RESULT
              result = _mm256_min_epi16(_mm256_max_epi16(result, min), max);
#endif
              _mm256_store_si256((__m256i *) a_ptr, result);
            }
            return true;
          }
        }
#else
        const T          value = B[0];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
        for (auto &x: m_out)
        {
          typename ComputationType<T>::type z = x;
          ComputationType<T>::quantize(z, shift);
          z += value;
          COUNTERS(z);
          SATURATE(z);
          x = static_cast<T>(z);
        }
#endif
      }
      else if (in[0]->dims().size() == 2)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << H << std::endl;
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
          {
            typename ComputationType<T>::type z = m_out(n, i);
            ComputationType<T>::quantize(z, shift);
            z += B[i];
            COUNTERS(z);
            SATURATE(z);
            m_out(n, i) = static_cast<T>(z);
          }
      }
      else if (in[0]->dims().size() == 3)
      {
        const Tensor<T> &B = *in[1];
        const int        N = in[0]->dims()[0];
        const int        H = in[0]->dims()[1];
        const int        W = in[0]->dims()[2];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << W << std::endl;
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
              typename ComputationType<T>::type z = m_out(n, i, j);
              ComputationType<T>::quantize(z, shift);
              z += B[j];
              COUNTERS(z);
              SATURATE(z);
              m_out(n, i, j) = static_cast<T>(z);
            }
      }
      else if (in[0]->dims().size() == 4)
      {
        const Tensor<T> &B = *in[1];
        const int        N = in[0]->dims()[0];
        const int        H = in[0]->dims()[1];
        const int        W = in[0]->dims()[2];
        const int        K = in[0]->dims()[3];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << K << std::endl;
#endif
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
              for (int k = 0; k < K; ++k)
              {
                typename ComputationType<T>::type z = m_out(n, i, j, k);
                ComputationType<T>::quantize(z, shift);
                z += B[k];
                COUNTERS(z);
                SATURATE(z);
                m_out(n, i, j, k) = static_cast<T>(z);
              }
      }
    }
  }
  else   // shift1
  {
    const int shift = shift1;
    if (in[0]->dims() == in[1]->dims())
    {
#if DEBUG_MODEL_ANALYZE
      std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
      for (auto it0 = m_out.begin(), it1 = in[1]->begin(); it0 != m_out.end(); ++it0, ++it1)
      {
        typename ComputationType<T>::type z = *it1;
        ComputationType<T>::quantize(z, shift);
        z += *it0;
        COUNTERS(z);
        SATURATE(z);
        *it0 = static_cast<T>(z);
      }
    }
    else
    {
      if (in[1]->size() == 1)
      {   // for constant
        const Tensor<T> &B    = *in[1];
#if __AVX2__
        if constexpr(std::is_same_v<T, int16_t>)
        {
          if (m_out.size() % 16 == 0) 
          {
            const __m256i     value = _mm256_srai_epi16(_mm256_set1_epi16(B[0]), shift);
            const __m256i     max   = _mm256_set1_epi16(32767);
            const __m256i     min   = _mm256_set1_epi16(-32768);
            T *a_ptr = m_out.data();
            for (int64_t i = 0; i < m_out.size(); i += 16, a_ptr+=16)
            {
              __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_ptr));
              __m256i result = _mm256_adds_epi16(x, value);
#if SATURATE_RESULT
              result = _mm256_min_epi16(_mm256_max_epi16(result, min), max);
#endif
              _mm256_store_si256((__m256i *) a_ptr, result);
            }
            return true;
          }
        }
#else
        T                valt = B[0];
        ComputationType<T>::quantize(valt, shift);
        const T value = valt;
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << m_out.size() << std::endl;
#endif
        for (auto &x: m_out)
        {
          typename ComputationType<T>::type z = x;
          z += value;
          COUNTERS(z);
          SATURATE(z);
          x = static_cast<T>(z);
        }
#endif
      }
      else if (in[0]->dims().size() == 2)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << H << std::endl;
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
          {
            typename ComputationType<T>::type z = B[i];
            ComputationType<T>::quantize(z, shift);
            z += m_out(n, i);
            COUNTERS(z);
            SATURATE(z);
            m_out(n, i) = static_cast<T>(z);
          }
      }
      else if (in[0]->dims().size() == 3)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        const int W = in[0]->dims()[2];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << W << std::endl;
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
            {
              typename ComputationType<T>::type z = B[j];
              ComputationType<T>::quantize(z, shift);
              z += m_out(n, i, j);
              COUNTERS(z);
              SATURATE(z);
              m_out(n, i, j) = static_cast<T>(z);
            }
      }
      else if (in[0]->dims().size() == 4)
      {
        const Tensor<T> &B = *in[1];
        assert(B.dims().size() == 1 || (B.dims().size() == 2 && B.dims()[0] == 1));
        const int N = in[0]->dims()[0];
        const int H = in[0]->dims()[1];
        const int W = in[0]->dims()[2];
        const int K = in[0]->dims()[3];
#if DEBUG_MODEL_ANALYZE
        std::cout << "\n[ANALYZE] add (in):\t" << K << std::endl;
#endif
        for (int n = 0; n < N; ++n)
          for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
              for (int k = 0; k < K; ++k)
              {
                typename ComputationType<T>::type z = B[k];
                ComputationType<T>::quantize(z, shift);
                z += m_out(n, i, j, k);
                COUNTERS(z);
                SATURATE(z);
                m_out(n, i, j, k) = static_cast<T>(z);
              }
      }
    }
  }
  return true;
}

// data in in[0]
// bias in in[1]
// assume data shape [N,W,H,D]
// assume bias shape [D]
template<typename T> bool Add<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);

  // either broadcast from a tensor of size [n] (use when input is Const) or [1,n]
  // of add if same dimensions
  if (in[1]->dims().size() == 1)
  {
    if (in[1]->dims()[0] != 1 && in[1]->dims()[0] != in[0]->dims().back())
      return false;
  }
  else if (in[1]->dims().size() == 2 && in[1]->dims()[0] == 1)
  {
    if (in[1]->dims()[1] != 1 && in[1]->dims()[1] != in[0]->dims().back())
      return false;
  }
  else
  {
    if (!(in[0]->dims() == in[1]->dims()))
      return false;
  }
  m_out.resize(in[0]->dims());
  m_initDone = true;
  return true;
}

template<typename T> bool Add<T>::loadInternal(std::istream &, Version) { return true; }

}   // namespace layers
}   // namespace sadl
