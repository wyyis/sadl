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
template<typename T> class PReLU : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;
  virtual bool mutateInput() const override { return true; }

protected:
  virtual bool loadInternal(std::istream &file, Version) override;
  bool prelu(std::vector<Tensor<T> *> &in);
#if __AVX2__ 
  bool prelu_simd256(std::vector<Tensor<T> *> &in);
#endif
#if __AVX512F__ || __AVX512BW__
  bool prelu_simd512(std::vector<Tensor<T> *> &in);
#endif
};

template<typename T> bool PReLU<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  assert(in[0]->dims() == m_out.dims());
#if __AVX512F__ 
  if ((typeid(T) == typeid(float)) && (in[0] -> size() % 16 == 0 && (in[1] -> size() == 1 || in[1] -> size() % 16 == 0)) )
    return prelu_simd512(in);
#endif
#if __AVX512BW__
  if ((typeid(T) == typeid(int16_t)) && (in[0] -> size() % 32 == 0 && (in[1] -> size() == 1 || in[1] -> size() % 32 == 0)) )
    return prelu_simd512(in);
#endif
#if __AVX2__
  if ((typeid(T) == typeid(float)) && (in[0] -> size() % 8 == 0 && (in[1] -> size() == 1 || in[1] -> size() % 8 == 0)) )
    return prelu_simd256(in);
  if ((typeid(T) == typeid(int16_t)) && (in[0] -> size() % 16 == 0 && (in[1] -> size() == 1 || in[1] -> size() % 16 == 0)) )
    return prelu_simd256(in);
#endif
  return prelu(in);
}


template<typename T> bool PReLU<T>::prelu(std::vector<Tensor<T> *> &in)//without simd
{
  const int in_N{ in[0]->dims()[0] };
  const int in_H{ in[0]->dims()[1] };
  const int in_W{ in[0]->dims()[2] };
  const int in_C{ in[0]->dims()[3] };

  const Tensor<T> &A = *in[1];
  swap(*in[0], m_out);
  // keep same qunatiz as input
  const int alpha_q = A.quantizer;
  for (int n_nb = 0; n_nb < in_N; n_nb++)
  {
    for (int c_nb = 0; c_nb < in_C; c_nb++)
    {
      // A.dims()[0] == 1, means all channels share the same alpha parameter
      const typename ComputationType<T>::type alpha = (A.dims()[0] == 1) ? A(0, 0, 0) : A(c_nb, 0, 0);
      for (int h_nb = 0; h_nb < in_H; h_nb++)
      {
        for (int w_nb = 0; w_nb < in_W; w_nb++)
        {
          if (m_out(n_nb, h_nb, w_nb, c_nb) < 0)
          {
            typename ComputationType<T>::type z = m_out(n_nb, h_nb, w_nb, c_nb) * alpha;
            ComputationType<T>::quantize(z, alpha_q);
            COUNTERS(z);
            COUNTERS_MAC(z);
            SATURATE(z);
            m_out(n_nb, h_nb, w_nb, c_nb) = z;
          } else {
            COUNTERS_MAC_NOP(1);
          } 
        }
      }
    }
  }
  return true;
}

#if __AVX2__
template<> 
inline bool PReLU<float>::prelu_simd256(std::vector<Tensor<float> *> &in)//simd256 float
{
  Tensor<float> &A = *in[1];
  swap(*in[0], m_out);
  float* const          data_ptr   = m_out.data();
  const float* const    alpha_ptr  = A.data();
  const __m256          m_zeros    = _mm256_setzero_ps();
  __m256                alpha      = _mm256_set1_ps(*A.data());
  for(int iter = 0; iter < m_out.size(); iter += 8)
  {
    if(A.size() > 1)
                        alpha      = _mm256_load_ps(alpha_ptr + iter % A.size());

    float* const        aptr       = data_ptr + iter;
    const __m256        a          = _mm256_load_ps(aptr);                 //load
    const __m256        min_a_zero = _mm256_min_ps(a,m_zeros);             //min(a,0)
    const __m256        max_a_zero = _mm256_max_ps(a,m_zeros);             //max(a,0)
    const __m256        b          = _mm256_mul_ps(min_a_zero,alpha);      //min(a,0)*alpha
    const __m256        v          = _mm256_add_ps(max_a_zero,b);          //max(a,0)+min(a,0)*alpha
   /*store*/                         _mm256_store_ps(aptr, v);
  }

  return true;
}


template<> 
inline bool PReLU<int16_t>::prelu_simd256(std::vector<Tensor<int16_t> *> &in)//simd256 int16 quantize
{
  Tensor<int16_t> &A = *in[1];
  swap(*in[0], m_out);
  int16_t* const        data_ptr   = m_out.data();
  const int16_t* const  alpha_ptr  = A.data();
  const __m256i         m_zeros    = _mm256_setzero_si256();
  __m256i               alpha      = _mm256_set1_epi16(*A.data());
  const int             alpha_q    = A.quantizer;
  for(int iter = 0; iter < m_out.size(); iter += 16)
  {
    if(A.size() > 1)
                        alpha      = _mm256_load_si256((__m256i*)(alpha_ptr + iter % A.size()));

    int16_t* const      aptr       = data_ptr + iter;
    const __m256i       a          = _mm256_load_si256((__m256i*)aptr);                //load
    const __m256i       min_a_zero = _mm256_min_epi16(a,m_zeros);                      //min(a,0)
    const __m256i       max_a_zero = _mm256_max_epi16(a,m_zeros);                      //max(a,0)
    const __m256i       lo         = _mm256_mullo_epi16(min_a_zero,alpha);             //min(a,0)*alpha lo part
    const __m256i       hi         = _mm256_mulhi_epi16(min_a_zero,alpha);             //min(a,0)*alpha hi part
    const __m256i       loq        = _mm256_srli_epi16(lo,alpha_q);                    //lo lo>>alpha_q logical shift
    const __m256i       hiq        = _mm256_slli_epi16(hi,16-alpha_q);                 //hi hi<<(16-alpha_q)
    const __m256i       q          = _mm256_or_si256(loq,hiq);                         //(hi part)|(lo part) use or operation
    const __m256i       v          = _mm256_add_epi16(max_a_zero,q);                   //max(a,0)+min(a,0)*alpha
    /*store*/                        _mm256_store_si256((__m256i*)aptr, v);
  }

  return true;
}


template<typename T> bool PReLU<T>::prelu_simd256(std::vector<Tensor<T> *> &in)//
{
  std::cerr << "[ERROR] simd type not supported: "  << std::endl;
  exit(-1);
}
#endif

#if __AVX512F__ 

template<> 
inline bool PReLU<float>::prelu_simd512(std::vector<Tensor<float> *> &in)//simd512 float
{
  Tensor<float> &A = *in[1];
  swap(*in[0], m_out);
  float* const          data_ptr   = m_out.data();
  const float* const    alpha_ptr  = A.data();
  const __m512          m_zeros    = _mm512_setzero_ps();
  __m512                alpha      = _mm512_set1_ps(*A.data());
  for(int iter = 0; iter < m_out.size(); iter += 16)
  {
    if(A.size() > 1)
                        alpha      = _mm512_load_ps(alpha_ptr + iter % A.size());

    float* const        aptr       = data_ptr + iter;                      //load
    const __m512        a          = _mm512_load_ps(aptr);
    const __m512        min_a_zero = _mm512_min_ps(a,m_zeros);             //min(a,0)
    const __m512        max_a_zero = _mm512_max_ps(a,m_zeros);             //max(a,0)
    const __m512        b          = _mm512_mul_ps(min_a_zero,alpha);      //min(a,0)*alpha
    const __m512        v          = _mm512_add_ps(max_a_zero,b);          //max(a,0)+min(a,0)*alpha
    /*store*/                        _mm512_store_ps(aptr, v);
  }

  return true;
}

#endif

#if __AVX512BW__ 
template<> 
inline bool PReLU<int16_t>::prelu_simd512(std::vector<Tensor<int16_t> *> &in)//simd512 int16 quantize
{
  Tensor<int16_t> &A = *in[1];
  swap(*in[0], m_out);
  int16_t* const        data_ptr   = m_out.data();
  const int16_t* const  alpha_ptr  = A.data();
  const __m512i         m_zeros    = _mm512_setzero_si512();
  __m512i               alpha      =_mm512_set1_epi16(*A.data());
  const int             alpha_q    = A.quantizer;
  for(int iter = 0; iter < m_out.size(); iter += 32)
  {
    if(A.size() > 1)
                        alpha      = _mm512_load_si512((__m512i*)(alpha_ptr + iter % A.size()));

    int16_t* const      aptr       = data_ptr + iter;
    const __m512i       a          = _mm512_load_si512((__m512i*)aptr);
    const __m512i       min_a_zero = _mm512_min_epi16(a,m_zeros);                      //min(a,0)
    const __m512i       max_a_zero = _mm512_max_epi16(a,m_zeros);                      //max(a,0)
    const __m512i       lo         = _mm512_mullo_epi16(min_a_zero,alpha);             //min(a,0)*alpha lo part
    const __m512i       hi         = _mm512_mulhi_epi16(min_a_zero,alpha);             //min(a,0)*alpha hi part
    const __m512i       loq        = _mm512_srli_epi16(lo,alpha_q);                    //lo lo>>alpha_q logical shift
    const __m512i       hiq        = _mm512_slli_epi16(hi,16-alpha_q);                 //hi hi<<(16-alpha_q)
    const __m512i       q          = _mm512_or_si512(hiq,loq);                         //(hi part)|(lo part) use or operation
    const __m512i       v          = _mm512_add_epi16(max_a_zero,q);                   //max(a,0)+min(a,0)*alpha
    /*store*/                        _mm512_store_si512((__m512i*)aptr, v);
  }

  return true;
}
#endif

#if __AVX512F__ || __AVX512BW__
template<typename T> 
bool PReLU<T>::prelu_simd512(std::vector<Tensor<T> *> &in)//
{
  std::cerr << "[ERROR] simd type not supported: "  << std::endl;
  exit(-1);
}
#endif

template<typename T> bool PReLU<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  m_out.resize(in[0]->dims());
  m_initDone = true;
  return true;
}

template<typename T> bool PReLU<T>::loadInternal(std::istream &, Version) { return true; }

}   // namespace layers
}   // namespace sadl
