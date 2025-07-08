/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2025, ITU/ISO/IEC
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
template<typename T> class Pad : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;

  enum pad_mode
  {
    pad_mode_constant = 0,
    pad_mode_reflect  = 1,
    pad_mode_edge     = 2
  };
  int                m_mode;
  std::array<int, 4> m_pads;   // [H_before, W_before, H_after, W_after]
  DUMP_MODEL_EXT;
};

template<typename T> bool Pad<T>::apply(std::vector<Tensor<T> *> &in)
{
#if DEBUG_SIMD
  std::cout << "\n[WARN] generic version pad" << std::endl;
#endif

  if (m_mode == pad_mode_constant)
  {
    assert(in.size() == 2);
  }
  else
  {
    assert(in.size() == 1);
  }
    
  m_out.quantizer     = in[0]->quantizer;
  const Tensor<T> &A  = *in[0];

  const auto &in_dims  = A.dims();
  const auto &out_dims = m_out.dims();

  const int N = out_dims[0];
  const int H = out_dims[1];
  const int W = out_dims[2];
  const int C = out_dims[3];

  const int in_H = in_dims[1];
  const int in_W = in_dims[2];

  const int pad_H_before = m_pads[0];
  const int pad_W_before = m_pads[1];

  for (int n = 0; n < N; ++n)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        for (int c = 0; c < C; ++c)
        {
          // Compute corresponding input coordinates (reverse mapping)
          int in_n = n;
          int in_h = h - pad_H_before;   // Ranges: [-pad_H_before, in_H + pad_H_after - 1]
          int in_w = w - pad_W_before;   // Negative: front padding, >=in_W: back padding
          int in_c = c;

          bool in_range = (in_h >= 0) && (in_h < in_H) && (in_w >= 0) && (in_w < in_W);

          if (m_mode == pad_mode_constant)
          {
            m_out(n, h, w, c) = in_range ? A(in_n, in_h, in_w, in_c) : (*in[1])[0];
          }
          else if (m_mode == pad_mode_reflect || m_mode == pad_mode_edge)
          {
            auto map_index = [&](int x, int limit) -> int
            {
              if (m_mode == pad_mode_edge)
              {
                return std::clamp(x, 0, limit - 1);
              }
              else
              {   // reflect mode
                if (limit <= 1)
                  return 0;
                x                = std::abs(x);
                const int period = 2 * (limit - 1);
                x %= period;
                return (x < limit) ? x : period - x;
              }
            };

            in_h               = map_index(in_h, in_H);
            in_w               = map_index(in_w, in_W);
            m_out(n, h, w, c)  = A(in_n, in_h, in_w, in_c);
          }
        }

  return true;
}

template<typename T> bool Pad<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (m_mode == pad_mode_constant)
  {
    if (in.size() != 2 || in[1]->dims().size() != 1)
      return false;   // constant mode requires a constant value as input
  }
  else
  {
    if (in.size() != 1)
      return false;
  }
  if (in[0]->dims().size() != 4)
    return false;

  Dimensions dims = in[0]->dims();

  // Reflect mode requires input H > pad_top and pad_bottom
  // and input W > pad_left and pad_right
  if (m_mode == pad_mode_reflect)
  {
    int pad_top    = m_pads[0];
    int pad_left   = m_pads[1];
    int pad_bottom = m_pads[2];
    int pad_right  = m_pads[3];

    int in_h = dims[1];
    int in_w = dims[2];

    if (in_h <= pad_top || in_h <= pad_bottom || in_w <= pad_left || in_w <= pad_right)
    {
      std::cerr << "[ERROR] Reflect mode requires input H > pad_{top,bottom}, W > pad_{left,right}" << std::endl;
      return false;
    }
  }

  dims[1]         = dims[1] + m_pads[0] + m_pads[2];   // H: before + after
  dims[2]         = dims[2] + m_pads[1] + m_pads[3];   // W: before + after

  m_out.resize(dims);
  m_initDone = true;
  return true;
}

template<typename T> bool Pad<T>::loadInternal(std::istream &file, Version)
{
  // Read padding mode (0: constant, 1: reflect, 2: edge)
  int32_t mode;
  file.read((char *) &mode, sizeof(mode));
  m_mode = mode;

  // Read 4 padding values: [H_before, W_before, H_after, W_after]
  for (int i = 0; i < 4; ++i)
  {
    int32_t pad;
    file.read((char *) &pad, sizeof(pad));

    if (pad < 0)
    {
      std::cerr << "[ERROR] Padding value must be >= 0, got: " << pad << std::endl;
      return false;
    }
    m_pads[i] = pad;
  }

  return true;
}

}   // namespace layers
}   // namespace sadl