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
template<typename T> class Softmax : public Layer<T>
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
  int32_t      m_axis;
  DUMP_MODEL_EXT;
};

template<typename T> bool Softmax<T>::apply(std::vector<Tensor<T> *> &in) {
  std::cerr << "[ERROR] Softmax version not defined" << std::endl;
  exit(-1);
  return false;
}

template<> inline bool Softmax<float>::apply(std::vector<Tensor<float> *> &in) {
  assert(in.size() == 1);
  assert(in[0]->size() == m_out.size());

  Dimensions d = m_out.dims();
  // resize done at init
  swapData(*in[0], m_out);
  
  if (m_axis == 0)
  {
    for (int c = 0; c < d[3]; c++)
    {
      for (int h = 0; h < d[1]; h++)
      {
        for (int w = 0; w < d[2]; w++)
        {
          float sum_axis = 0.0;
          for (int n = 0; n < d[0]; n++)
          {
            m_out(n, h, w, c) = std::exp(m_out(n, h, w, c));
            sum_axis += m_out(n, h, w, c);
          }
          for (int n = 0; n < d[0]; n++)
          {
            m_out(n, h, w, c) = m_out(n, h, w, c) / sum_axis;
          }
        }
      }
    }
      
  }
  else if (m_axis == 1)
  {
    for (int n = 0; n < d[0]; n++)
    {
      for (int c = 0; c < d[3]; c++)
      {
        for (int w = 0; w < d[2]; w++)
        {
          float sum_axis = 0.0;
          for (int h = 0; h < d[1]; h++)
          {
            m_out(n, h, w, c) = std::exp(m_out(n, h, w, c));
            sum_axis += m_out(n, h, w, c);
          }
          for (int h = 0; h < d[1]; h++)
          {
            m_out(n, h, w, c) = m_out(n, h, w, c) / sum_axis;
          }
        }
      }
    }
  }
  else if (m_axis == 2)
  {
    for (int n = 0; n < d[0]; n++)
    {
      for (int c = 0; c < d[3]; c++)
      {
        for (int h = 0; h < d[1]; h++)
        {
          float sum_axis = 0.0;
          for (int w = 0; w < d[2]; w++)
          {
            m_out(n, h, w, c) = std::exp(m_out(n, h, w, c));
            sum_axis += m_out(n, h, w, c);
          }
          for (int w = 0; w < d[2]; w++)
          {
            m_out(n, h, w, c) = m_out(n, h, w, c) / sum_axis;
          }
        }
      }
    }
  }
  else if (m_axis == 3)
  {
    for (int n = 0; n < d[0]; n++)
    {
      for (int h = 0; h < d[1]; h++)
      {
        for (int w = 0; w < d[2]; w++)
        {
          float sum_axis = 0.0;
          for (int c = 0; c < d[3]; c++)
          {
            m_out(n, h, w, c) = std::exp(m_out(n, h, w, c));
            sum_axis += m_out(n, h, w, c);
          }
          for (int c = 0; c < d[3]; c++)
          {
            m_out(n, h, w, c) = m_out(n, h, w, c) / sum_axis;
          }
        }
      }
    }
      
  }
  else
  {
    std::cerr << "[ERROR] invalid axis: " << m_axis << std::endl;
    return false;
  }

  return true;
}

template<typename T> bool Softmax<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 1)
    return false;
 
  m_out.resize(in[0]->dims());
  m_initDone = true;
  return true;
}

template<typename T> bool Softmax<T>::loadInternal(std::istream &file, Version)
{
  // load values
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  if (x < 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid axis: " << x << std::endl;
    return false;
  }
  m_axis = x;
  SADL_DBG(std::cout << "  - axis: " << m_axis << std::endl);
  return true;
}

}   // namespace layers
}   // namespace sadl
