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
template<typename T> class Slice : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
  int32_t      m_start_d;
  int32_t      m_end_d;
  DUMP_MODEL_EXT;
};

template<typename T> bool Slice<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 1);
  const Tensor<T> &A = *in[0];
  int              in_H{ A.dims()[1] };
  int              in_W{ A.dims()[2] };
  const int        in_D{ A.dims()[3] };
  constexpr int    im_nb   = 0;
  constexpr int    pow2_31 = std::numeric_limits<int>::max();
  // ONNX is sending 2^31 - 1 as value if end index is last channel
  if (m_end_d == pow2_31)
  {
    m_end_d = in_D;
  }

  m_out.quantizer = A.quantizer;

  for (int im_i = 0; im_i < in_H; im_i++)
  {
    for (int im_j = 0; im_j < in_W; im_j++)
    {
      for (int im_d = m_start_d; im_d < m_end_d; im_d++)
      {
        m_out(im_nb, im_i, im_j, im_d - m_start_d) = A(im_nb, im_i, im_j, im_d);
      }
    }
  }
  return true;
}

template<typename T> bool Slice<T>::init(const std::vector<Tensor<T> *> &in)
{
  /*
  Right now slicing only across the depth
  */
  constexpr int pow2_31 = std::numeric_limits<int>::max();
  if (in.size() != 1)
    return false;
  SADL_DBG(std::cout << " - input Slice " << in[0]->dims() << std::endl);

  Dimensions dim;
  dim.resize(4);
  dim[0] = in[0]->dims()[0];
  dim[1] = in[0]->dims()[1];
  dim[2] = in[0]->dims()[2];
  // ONNX is sending 2^31 - 1 as value if end index is last channel
  if (m_end_d == pow2_31)
  {
    m_end_d = in[0]->dims()[3];
  }
  dim[3] = m_end_d - m_start_d;
  m_out.resize(dim);
  SADL_DBG(std::cout << "  - output Slice: " << m_out.dims() << std::endl);

  m_initDone = true;
  return true;
}

template<typename T> bool Slice<T>::loadInternal(std::istream &file, Version /*v*/)
{
  file.read((char *) &m_start_d, sizeof(m_start_d));
  SADL_DBG(std::cout << "  - start_d: " << m_start_d << std::endl);

  file.read((char *) &m_end_d, sizeof(m_end_d));
  SADL_DBG(std::cout << "  - end_d: " << m_end_d << std::endl);

  return true;
}

}   // namespace layers
}   // namespace sadl
