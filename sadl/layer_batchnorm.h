/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2024, ITU/ISO/IEC
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
template<typename T> class BatchNorm : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  using T2 = typename ComputationType<T>::type;
  T2           bitwise_sqrt(T2 x, int q);
  Tensor<T2>   std_var;
  virtual bool loadInternal(std::istream &file, Version v) override;
};

template<typename T> bool BatchNorm<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 6);
  assert(in[0]->dims() == m_out.dims());

  const Tensor<T> &A      = *in[0];
  const Tensor<T> &weight = *in[1];
  const Tensor<T> &bias   = *in[2];
  const Tensor<T> &mean   = *in[3];
  const Tensor<T> &var    = *in[4];
  const Tensor<T> &eps    = *in[5];
  const int        N      = m_out.dims()[0];
  const int        H      = m_out.dims()[1];
  const int        W      = m_out.dims()[2];
  const int        D      = m_out.dims()[3];

  m_out.quantizer   = in[0]->quantizer;
  m_out.border_skip = in[0]->border_skip;
  int q             = A.quantizer;

  // pre-compute standard variance
  for (int im_d = 0; im_d < D; ++im_d)
  {
    if constexpr (std::is_same<T, float>::value)
      std_var(im_d) = sqrt(var(im_d) + eps(0));
    else
      std_var(im_d) = bitwise_sqrt(var(im_d) + eps(0), q);
  }

  for (int im_nb = 0; im_nb < N; ++im_nb)
  {
    for (int im_i = 0; im_i < H; ++im_i)
    {
      for (int im_j = 0; im_j < W; ++im_j)
      {
        for (int im_d = 0; im_d < D; ++im_d)
        {
          T2 num = (A(im_nb, im_i, im_j, im_d) - mean(im_d)) * weight(im_d);
          ComputationType<T>::quantize(num, q);

          if constexpr (std::is_same<T, float>::value)
          {
            num = num / std_var(im_d);
          }
          else
          {
            // floor(a + b / 2) / b == round(a / b)
            int a = 1 << q;
            int b  = static_cast<int>(std_var(im_d));
            int result = (a + b / 2) / b;
            num = num * result;
          }

          num += bias(im_d);
          COUNTERS(num);
          COUNTERS_MAC(num);
          SATURATE(num);
          m_out(im_nb, im_i, im_j, im_d) = static_cast<T>(num);
        }
      }
    }
  }

  return true;
}

template<typename T> typename BatchNorm<T>::T2 BatchNorm<T>::bitwise_sqrt(typename BatchNorm<T>::T2 x, int q)
{
  /*
   * This function computes the square root of x as an integer type.
   * Taking T as int16 as an example:
   * When q is even, shift_left_bits=16, shift_right_bits=(16 - q) / 2
   * When q is odd, shift_left_bits=16+1, shift_right_bits=(16 + 1 - q) / 2 (adding 1 to make shift_right_bits an integer)
   * Convert x to the ComputationType<T>::type type, then left shift by shift_left_bits, and compute the square root.
   * After computation, right shift the result by shift_right_bits.
   */

  static_assert(std::is_integral<T>::value, "T must be an integer type!");
  assert(x >= 0);

  using T2 = typename BatchNorm<T>::T2;
  T2 result = 0;

  // Shift
  int shift_left_bits;
  if (q % 2)
    shift_left_bits = sizeof(T2) * 8 / 2 + 1;
  else
    shift_left_bits = sizeof(T2) * 8 / 2;
  T2 num = x;
  ComputationType<T>::shift_left(num, shift_left_bits);

  // Square root
  T2 bit = T2(1) << (sizeof(T2) * 8 / 2 - 1); // initial bit=2^15
  while (bit > 0)
  {
    T2 temp = result | bit;
    if (temp * temp <= num)
      result = temp;
    bit >>= 1;   // Move to the next lower bit
  }

  // Shift back
  int shift_right_bits;
  if (q % 2)
    shift_right_bits = (sizeof(T2) * 8 / 2 + 1 - q) / 2;
  else
    shift_right_bits = (sizeof(T2) * 8 / 2 - q) / 2;
  T2 out = result;
  ComputationType<T>::quantize(out, shift_right_bits);

  return out;
}

template<typename T> bool BatchNorm<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 6)
    return false;
  SADL_DBG(std::cout << "  - input BatchNorm: " << in[0]->dims() << std::endl);
  SADL_DBG(std::cout << "  - weight:          " << in[1]->dims() << std::endl);
  SADL_DBG(std::cout << "  - bias:            " << in[2]->dims() << std::endl);
  SADL_DBG(std::cout << "  - running_mean:    " << in[3]->dims() << std::endl);
  SADL_DBG(std::cout << "  - running_var:     " << in[4]->dims() << std::endl);
  SADL_DBG(std::cout << "  - epsilon:         " << in[5]->dims() << std::endl);

  int m_s = in[0]->dims().size();
  if (in[0]->dims()[m_s - 1] != in[1]->dims()[0])
  {
    std::cerr << "[ERROR] invalid weight size: " << in[1]->dims() << std::endl;
    return false;
  }
  if (in[0]->dims()[m_s - 1] != in[2]->dims()[0])
  {
    std::cerr << "[ERROR] invalid bias size: " << in[2]->dims() << std::endl;
    return false;
  }
  if (in[0]->dims()[m_s - 1] != in[3]->dims()[0])
  {
    std::cerr << "[ERROR] invalid running_mean size: " << in[3]->dims() << std::endl;
    return false;
  }
  if (in[0]->dims()[m_s - 1] != in[4]->dims()[0])
  {
    std::cerr << "[ERROR] invalid running_var size: " << in[4]->dims() << std::endl;
    return false;
  }
  if (in[5]->dims()[0] != 1)
  {
    std::cerr << "[ERROR] invalid epsilon size: " << in[5]->dims() << std::endl;
    return false;
  }

  Tensor<T> &eps = *in[5];
  if (eps(0) == 0)
  {
    if constexpr (std::is_same<T, float>::value)
      eps(0) = 1e-5f;
    else
      eps(0) = 1;
  }

  std_var.resize(in[4]->dims());
  m_out.resize(in[0]->dims());
  m_initDone = true;
  
  return true;
}

template<typename T> bool BatchNorm<T>::loadInternal(std::istream &file, Version v) { return true; }

}   // namespace layers
}   // namespace sadl
