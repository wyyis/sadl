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
};

template<typename T> bool PReLU<T>::apply(std::vector<Tensor<T> *> &in)
{
  const int in_N{ in[0]->dims()[0] };
  const int in_H{ in[0]->dims()[1] };
  const int in_W{ in[0]->dims()[2] };
  const int in_C{ in[0]->dims()[3] };

  assert(in.size() == 2);
  assert(in[0]->dims() == m_out.dims());
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
            m_out(n_nb, h_nb, w_nb, c_nb) = z;
          }
        }
      }
    }
  }
  return true;
}

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
