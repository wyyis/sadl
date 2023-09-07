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
template<typename T> class GridSample : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version) override;
  void         gs_denormalize(float &x, int length);
  void         pixel_addr_at_grid(int &addr, int H, int W, int C, int i, int j, int c);
  void         atomic_bilinear(const Tensor<T> &X, float x, float y, int shift, int addr_out);
  int          m_align_corners;   // 0: False, 1: True
  int          m_mode;            // 0: "nearest", 1: "bilinear"
  int          m_padding_mode;    // 0: "border"
  using T2 = typename ComputationType<T>::type;
  DUMP_MODEL_EXT;
};

template<typename T> bool GridSample<T>::loadInternal(std::istream &file, Version)
{
  if (!std::is_same<T, float>::value)
  {
    std::cerr << "[ERROR] Currently, GridSample only supports float data type." << std::endl;
    return false;
  }

  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  m_align_corners = x;
  SADL_DBG(std::cout << "  - align_corners: " << m_align_corners << std::endl);
  file.read((char *) &x, sizeof(x));
  m_mode = x;
  SADL_DBG(std::cout << "  - mode: " << m_mode << std::endl);
  file.read((char *) &x, sizeof(x));
  m_padding_mode = x;
  SADL_DBG(std::cout << "  - padding_mode: " << m_padding_mode << std::endl);
  if (m_mode != 0 && m_mode != 1)
  {
    std::cerr << "[ERROR] invalid mode: " << m_mode << std::endl;
    return false;
  }
  if (m_padding_mode != 0)
  {
    std::cerr << "[ERROR] invalid padding mode: " << m_padding_mode << std::endl;
    return false;
  }
  return true;
}

template<typename T> bool GridSample<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  Dimensions dim;
  dim.resize(4);
  dim[0] = in[0]->dims()[0];   // N, inherit from X
  dim[1] = in[1]->dims()[3];   // H_out, inherit from grid
  dim[2] = in[1]->dims()[1];   // W_out, inherit from grid
  dim[3] = in[0]->dims()[3];   // C, inherit from X
  m_out.resize(dim);
  m_initDone = true;
  return true;
}

template<typename T> bool GridSample<T>::apply(std::vector<Tensor<T> *> &in)
{
  const Tensor<T> &X    = *in[0];   // (1,H_in,W_in,C)
  const Tensor<T> &grid = *in[1];   // (1,W_out,2,H_out)
  m_out.quantizer       = X.quantizer;
  assert(X.dims()[0] == 1 && grid.dims()[0] == 1 && grid.dims()[2] == 2);

  int   C                    = X.dims()[3];
  int   H_in                 = X.dims()[1];
  int   W_in                 = X.dims()[2];
  int   H_out                = grid.dims()[3];
  int   W_out                = grid.dims()[1];
  float x_min                = (m_align_corners) ? 0 : -0.5;
  float x_max                = (m_align_corners) ? W_in - 1 : W_in - 0.5;
  float y_min                = (m_align_corners) ? 0 : -0.5;
  float y_max                = (m_align_corners) ? H_in - 1 : H_in - 0.5;
  bool  normalize_grid_flag  = !std::is_same<T, float>::value;
  float normalize_grid_coeff = 2 << (grid.quantizer - 1);
  int   shift                = grid.quantizer;

  // Given an input X and a flow-field grid, computes the output Y using X values and pixel locations from the grid.
  int addr_grid_x = -W_out, addr_grid_y = -W_out + H_out;
  for (int im_j = 0; im_j < W_out; im_j++)
  {
    addr_grid_x += H_out;
    addr_grid_y += H_out;
    for (int im_i = 0; im_i < H_out; im_i++)
    {
      // compute pixel locations from the grid
      float x = grid[addr_grid_x++];
      float y = grid[addr_grid_y++];
      if (normalize_grid_flag)
      {
        x = x / normalize_grid_coeff;
        y = y / normalize_grid_coeff;
      }
      gs_denormalize(x, W_in);
      gs_denormalize(y, H_in);
      if (m_mode == 0)   // nearest
      {
        x = round(x);
        y = round(y);
      }
      if (x < x_min || x > x_max || y < y_min || y > y_max)
      {
        if (m_padding_mode == 0)   // border
        {
          x = (x < 0) ? 0 : ((x > W_in - 1) ? W_in - 1 : x);
          y = (y < 0) ? 0 : ((y > H_in - 1) ? H_in - 1 : y);
        }
      }
      // compute the output Y using X values and pixel locations
      int addr_out = (im_i * W_out + im_j) * C;
      if (m_mode == 0)   // nearest
      {
        int addr_grid = 0;
        pixel_addr_at_grid(addr_grid, H_in, W_in, C, int(y), int(x), 0);
        for (int im_c = 0; im_c < C; im_c++)
        {
          m_out[addr_out++] = X[addr_grid];   // same data type
          COUNTERS(X[addr_grid++]);
        }
      }
      else if (m_mode == 1)   // bilinear
      {
        atomic_bilinear(X, x, y, shift, addr_out);
      }
    }
  }
  return true;
}

template<typename T> void GridSample<T>::gs_denormalize(float &x, int length)
{
  if (m_align_corners)
  {
    x = (x + 1) / 2.0 * (length - 1);
  }
  else
  {
    x = ((x + 1) * length - 1) / 2.0;
  }
}

template<typename T> void GridSample<T>::pixel_addr_at_grid(int &addr, int H, int W, int C, int i, int j, int c)
{
  if (m_padding_mode == 0)   // border
  {
    i    = (i < 0) ? 0 : ((i > H - 1) ? H - 1 : i);
    j    = (j < 0) ? 0 : ((j > W - 1) ? W - 1 : j);
    addr = C * (W * i + j) + c;   // (0, i, j, c)
  }
}

template<typename T> void GridSample<T>::atomic_bilinear(const Tensor<T> &X, float x, float y, int shift, int addr_out)
{
  int H_in = X.dims()[1];
  int W_in = X.dims()[2];
  int C    = X.dims()[3];

  T2    num{};
  T     coeff11{}, coeff12{}, coeff21{}, coeff22{};
  float x1 = floor(x), y1 = floor(y), x2 = x1 + 1, y2 = y1 + 1;
  float normalize_grid_coeff = 2 << (shift - 1);
  if (std::is_same<T2, float>::value)
  {
    coeff11 = (y2 - y) * (x2 - x);   // dy2 * dx2
    coeff12 = (y2 - y) * (x - x1);   // dy2 * dx1
    coeff21 = (y - y1) * (x2 - x);   // dy1 * dx2
    coeff22 = (y - y1) * (x - x1);   // dy1 * dx1
  }
  else
  {
    coeff11 = (y2 - y) * (x2 - x) * normalize_grid_coeff;
    coeff12 = (y2 - y) * (x - x1) * normalize_grid_coeff;
    coeff21 = (y - y1) * (x2 - x) * normalize_grid_coeff;
    coeff22 = (y - y1) * (x - x1) * normalize_grid_coeff;
  }

  int addr_grid11 = 0, addr_grid12 = 0, addr_grid21 = 0, addr_grid22 = 0;
  pixel_addr_at_grid(addr_grid11, H_in, W_in, C, y1, x1, 0);
  pixel_addr_at_grid(addr_grid12, H_in, W_in, C, y1, x2, 0);
  pixel_addr_at_grid(addr_grid21, H_in, W_in, C, y2, x1, 0);
  pixel_addr_at_grid(addr_grid22, H_in, W_in, C, y2, x2, 0);
  for (int im_c = 0; im_c < C; im_c++)
  {
    num = coeff11 * X[addr_grid11++] + coeff12 * X[addr_grid12++] + coeff21 * X[addr_grid21++] + coeff22 * X[addr_grid22++];
    ComputationType<T>::quantize(num, shift);
    {
      COUNTERS_MAC(coeff11);
      COUNTERS_MAC(coeff12);
      COUNTERS_MAC(coeff21);
      COUNTERS_MAC(coeff22);
    }
    COUNTERS(num);
    SATURATE(num);
    m_out[addr_out++] = static_cast<T>(num);
  }
}

}   // namespace layers
}   // namespace sadl
