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
#include "interpolation_utils.h"

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
  using T2 = typename ComputationType<T>::type;
  virtual bool loadInternal(std::istream &file, Version) override;
  bool         gridsample_nearest(std::vector<Tensor<T> *> &in);
  bool         gridsample_bilinear(std::vector<Tensor<T> *> &in);
  void         get_bilinear_coeffs(T2 y, T2 x, int pos[], T2 coeffs[]);
  void         gs_denormalize(T2 &x, int length);
  void         pixel_addr_at_grid(int H, int W, int &i, int &j);

  enum gridsample_mode
  {
    gridsample_mode_nearest  = 0,
    gridsample_mode_bilinear = 1
  };
  enum gridsample_paddingmode
  {
    gridsample_paddingmode_border = 0
  };
  int m_align_corners;   // 0: False, 1: True
  int m_mode;            // 0: "nearest", 1: "bilinear"
  int m_padding_mode;    // 0: "border"
  int m_grid_quantizer;
  DUMP_MODEL_EXT;
};

template<typename T> bool GridSample<T>::loadInternal(std::istream &file, Version)
{
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
  if (m_align_corners != 1)
  {
    std::cerr << "[ERROR] invalid align corners: " << m_align_corners << std::endl;
    return false;
  }
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
  dim[0] = in[0]->dims()[0];   // N, inherit from data
  dim[1] = in[1]->dims()[3];   // H_out, inherit from grid
  dim[2] = in[1]->dims()[1];   // W_out, inherit from grid
  dim[3] = in[0]->dims()[3];   // C, inherit from data
  m_out.resize(dim);
  m_initDone = true;
  return true;
}

template<typename T> bool GridSample<T>::apply(std::vector<Tensor<T> *> &in)
{
  if (m_mode == gridsample_mode_nearest)
    return gridsample_nearest(in);
  else if (m_mode == gridsample_mode_bilinear)
    return gridsample_bilinear(in);

  return false;
}

template<typename T> bool GridSample<T>::gridsample_nearest(std::vector<Tensor<T> *> &in)
{
  // Given an input data and a flow-field grid, computes the output Y using data
  // values and pixel locations from the grid.
  const Tensor<T> &data = *in[0];   // (1,H_in,W_in,C)
  const Tensor<T> &grid = *in[1];   // (1,W_out,2,H_out)
  m_out.quantizer       = data.quantizer;
  m_grid_quantizer      = grid.quantizer;
  assert(data.dims()[0] == 1 && grid.dims()[0] == 1 && grid.dims()[2] == 2);

  constexpr int im_nb = 0;
  int           H_in  = data.dims()[1];
  int           W_in  = data.dims()[2];
  int           H_out = grid.dims()[3];
  int           W_out = grid.dims()[1];

  for (int im_j = 0; im_j < W_out; im_j++)
  {
    for (int im_i = 0; im_i < H_out; im_i++)
    {
      // compute pixel locations from the grid
      T2 x = grid(im_nb, im_j, 0, im_i);
      T2 y = grid(im_nb, im_j, 1, im_i);
      gs_denormalize(x, W_in);
      gs_denormalize(y, H_in);

      // compute the output Y using data values and pixel locations
      int x_int = int(x);
      int y_int = int(y);
      pixel_addr_at_grid(H_in, W_in, y_int, x_int);

      nearest_in_channels(data, im_i, im_j, y_int, x_int, m_out);
    }
  }
  return true;
}

template<typename T> bool GridSample<T>::gridsample_bilinear(std::vector<Tensor<T> *> &in)
{
  // Given an input data and a flow-field grid, computes the output Y using data
  // values and pixel locations from the grid.
  const Tensor<T> &data = *in[0];   // (1,H_in,W_in,C)
  const Tensor<T> &grid = *in[1];   // (1,W_out,2,H_out)
  m_out.quantizer       = data.quantizer;
  m_grid_quantizer      = grid.quantizer;
  assert(data.dims()[0] == 1 && grid.dims()[0] == 1 && grid.dims()[2] == 2);

  constexpr int im_nb = 0;
  int           shift = m_grid_quantizer + 1;
  int           H_in  = data.dims()[1];
  int           W_in  = data.dims()[2];
  int           H_out = grid.dims()[3];
  int           W_out = grid.dims()[1];

  for (int im_j = 0; im_j < W_out; im_j++)
  {
    for (int im_i = 0; im_i < H_out; im_i++)
    {
      // compute pixel locations from the grid
      T2 x = grid(im_nb, im_j, 0, im_i);
      T2 y = grid(im_nb, im_j, 1, im_i);
      gs_denormalize(x, W_in);
      gs_denormalize(y, H_in);

      // compute the output Y using data values and pixel locations
      T2   coeffs[4];
      int  pos[4];
      int &x_ori_left = pos[0], &y_ori_top = pos[1], &x_ori_right = pos[2], &y_ori_bottom = pos[3];

      get_bilinear_coeffs(y, x, pos, coeffs);
      pixel_addr_at_grid(H_in, W_in, y_ori_top, x_ori_left);
      pixel_addr_at_grid(H_in, W_in, y_ori_top, x_ori_right);
      pixel_addr_at_grid(H_in, W_in, y_ori_bottom, x_ori_left);
      pixel_addr_at_grid(H_in, W_in, y_ori_bottom, x_ori_right);

      BILINEAR_COUNTERS(data, coeffs);
      bilinear_in_channels(data, coeffs, pos, shift, im_i, im_j, m_out);
    }
  }
  return true;
}

template<> inline void GridSample<float>::gs_denormalize(float &x, int length)
{
  if (m_mode == gridsample_mode_nearest)
  {
    x = (x + 1) * (length - 1) / 2.0f;
    x = round(x);
  }
  else if (m_mode == gridsample_mode_bilinear)
  {
    x = (x + 1) * (length - 1) / 2.0f;
  }
}

template<typename T> void GridSample<T>::gs_denormalize(T2 &x, int length)
{
  int shift = m_grid_quantizer;
  if (m_mode == gridsample_mode_nearest)
  {
    T2 normalize_bias = (T)(1 << shift);
    x                 = (x + normalize_bias) * (length - 1);
    x                 = (x + normalize_bias) >> (shift + 1);   // round
  }
  else if (m_mode == gridsample_mode_bilinear)
  {
    T2 normalize_bias = (T)(1 << shift);
    x                 = (x + normalize_bias) * (length - 1);   // shift later to to maintain precision in higher-order bits during computation
  }
}

template<> inline void GridSample<float>::get_bilinear_coeffs(float y, float x, int pos[], float coeffs[])
{
  float &coeff11 = coeffs[0], &coeff12 = coeffs[1], &coeff21 = coeffs[2], &coeff22 = coeffs[3];
  int   &x_ori_left = pos[0], &y_ori_top = pos[1], &x_ori_right = pos[2], &y_ori_bottom = pos[3];

  x_ori_left   = (int)floor(x);
  y_ori_top    = (int)floor(y);
  x_ori_right  = x_ori_left + 1;
  y_ori_bottom = y_ori_top + 1;
  float dy2    = y_ori_bottom - y;
  float dy1    = y - y_ori_top;
  float dx2    = x_ori_right - x;
  float dx1    = x - x_ori_left;
  coeff11      = dy2 * dx2;
  coeff12      = dy2 * dx1;
  coeff21      = dy1 * dx2;
  coeff22      = dy1 * dx1;
}

template<typename T> void GridSample<T>::get_bilinear_coeffs(T2 y, T2 x, int pos[], T2 coeffs[])
{
  T2  &coeff11 = coeffs[0], &coeff12 = coeffs[1], &coeff21 = coeffs[2], &coeff22 = coeffs[3];
  int &x_ori_left = pos[0], &y_ori_top = pos[1], &x_ori_right = pos[2], &y_ori_bottom = pos[3];

  int shift    = m_grid_quantizer + 1;
  x_ori_left   = (int) (x >> shift);   // floor
  y_ori_top    = (int) (y >> shift);
  x_ori_right  = x_ori_left + 1;
  y_ori_bottom = y_ori_top + 1;
  T2 dy2       = (y_ori_bottom << shift) - y;
  T2 dy1       = y - (y_ori_top << shift);
  T2 dx2       = (x_ori_right << shift) - x;
  T2 dx1       = x - (x_ori_left << shift);
  coeff11      = (dy2 * dx2) >> shift;
  coeff12      = (dy2 * dx1) >> shift;
  coeff21      = (dy1 * dx2) >> shift;
  coeff22      = (dy1 * dx1) >> shift;
}

template<typename T> void GridSample<T>::pixel_addr_at_grid(int H, int W, int &i, int &j)
{
  if (m_padding_mode == gridsample_paddingmode_border)
  {
    i = (i < 0) ? 0 : ((i > H - 1) ? H - 1 : i);
    j = (j < 0) ? 0 : ((j > W - 1) ? W - 1 : j);
  }
}

}   // namespace layers
}   // namespace sadl
