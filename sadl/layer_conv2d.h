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
#include <cmath>
#if __AVX2__
#include <immintrin.h>
#endif
#include "simd_utils.h"

namespace sadl
{
namespace layers
{
template<typename T> class Conv2D : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;   // to avoid this->
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version /*v*/) override;
  Dimensions   m_strides;
  Dimensions   m_pads;
  int          m_q      = 0;
  int          m_groups = 1;

  template<int s_h, int s_w> bool apply_s(const Tensor<T> &A, const Tensor<T> &kernel);

  // should never be used
  void conv2d(const Tensor<T> &A, const Tensor<T> &kernel);
  
  template<int s_h, int s_w> bool conv2d_core(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int n> void multiply_add_n_points(const T* input, const T coeff, typename ComputationType<T>::type* sum);
  void simd_multiply_add_16_points(const T* input, const T coeff, typename ComputationType<T>::type* sum);


  template<int s_h, int s_w> void conv2d_5x5_s(const Tensor<T> &A, const Tensor<T> &kernel);
  template<int s_h, int s_w> void conv2d_2x2_s(const Tensor<T> &A, const Tensor<T> &kernel);

  // 1x1
  template<int s_h, int s_w> void conv2d_1x1_s_dispatch(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int in_D, int s_h, int s_w> void conv2d_1x1_s_d(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w> void conv2d_1x1_s(const Tensor<T> &A, const Tensor<T> &kernel);

  // 3x3
  template<int s_h, int s_w> void conv2d_3x3_s_peel(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w> void conv2d_3x3_s_core_dispatch(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w> void conv2d_3x3_s_core(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int in_D, int s_h, int s_w> void conv2d_3x3_s_d_core(const Tensor<T> &A, const Tensor<T> &kernel);

  // i x j
  template<int s_h, int s_w> void conv2d_ixj_s_peel(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w> void conv2d_ixj_s_core_dispatch(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w> void conv2d_ixj_s_core(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int in_D, int s_h, int s_w> void conv2d_ixj_s11_gD_d_core(const Tensor<T> &A, const Tensor<T> &kernel);

  template<int in_D, int ihalf_size, int jhalf_size> void conv2d_ixj_s11_g1_d_core(const Tensor<T> &A, const Tensor<T> &kernel);

#if __AVX2__
  template<int in_D, int s_h, int s_w> void simd8_conv2d_1x1_s_d(const Tensor<T> & /*A*/, const Tensor<T> & /*kernel*/)
  {
    assert(false);
    exit(-1);
  }
  template<int in_D, int s_h, int s_w> void simd16_conv2d_1x1_s_d(const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd8_conv2d_1x1_s_d<in_D, s_h, s_w>(A, kernel);
  }
  template<int in_D, int s_h, int s_w> void simd32_conv2d_1x1_s_d(const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd16_conv2d_1x1_s_d<in_D, s_h, s_w>(A, kernel);
  }

  template<int in_D, int s_h, int s_w> void simd8_conv2d_3x3_s_d(const Tensor<T> & /*A*/, const Tensor<T> & /*kernel*/)
  {
    assert(false);
    exit(-1);
  }
  template<int in_D, int s_h, int s_w> void simd16_conv2d_3x3_s_d(const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd8_conv2d_3x3_s_d<in_D, s_h, s_w>(A, kernel);
  }
  template<int in_D, int s_h, int s_w> void simd32_conv2d_3x3_s_d(const Tensor<T> &A, const Tensor<T> &kernel)
  {
    simd16_conv2d_3x3_s_d<in_D, s_h, s_w>(A, kernel);
  }
  template<int in_D, int ihalf_size, int jhalf_size> void simd16_conv2d_ixj_s11_g1_d_core(const Tensor<T> &A, const Tensor<T> &kernel) {
    conv2d_ixj_s11_g1_d_core<in_D,ihalf_size,jhalf_size>(A,kernel);

  }

  template<int in_D, int ihalf_size, int jhalf_size> void simd32_conv2d_ixj_s11_g1_d_core(const Tensor<T> &A, const Tensor<T> &kernel) {
    simd16_conv2d_ixj_s11_g1_d_core<in_D,ihalf_size,jhalf_size>(A,kernel);
  }
#endif
  static constexpr int bufSize = 256 + 8 * 2;
  static constexpr int vectorSize = 16; // has to be 16 for the current SIMD optimization
  static constexpr int dSize = 7;
  T input_tempo[dSize][bufSize][bufSize + vectorSize]; 

  DUMP_MODEL_EXT;
};

// assume data in in[0] and kernel in in[1]
// data [batch, in_height, in_width, in_channels]
// kernel [filter_height, filter_width, in_channels, out_channels]
template<typename T> bool Conv2D<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  assert(in[0]->dims().size() == 4);
  assert(in[1]->dims().size() == 4);
  const Tensor<T> &A      = *in[0];
  const Tensor<T> &kernel = *in[1];
  m_out.quantizer          = A.quantizer - m_q;
  m_out.border_skip        = A.border_skip;

#if DEBUG_MODEL_ANALYZE
  std::cout << "\n[ANALYZE] conv (in/out):\t"<<A.dims()[3]<<'\t'<<kernel.dims()[2]<<std::endl;
#endif
  assert(m_out.quantizer >= 0);
  assert(kernel.quantizer + m_q >= 0);
#define DEBUG_CONV2D 0
#if DEBUG_CONV2D
  std::cout << "[WARNING] generic conv2d for debug" << std::endl;
  conv2d(A, kernel);
  return true;
#endif
  if (m_strides[1] == 1 && m_strides[2] == 1)
  {
    return apply_s<1, 1>(A, kernel);
  }
  else if (m_strides[1] == 1 && m_strides[2] == 2)
  {
    return apply_s<1, 2>(A, kernel);
  }
  else if (m_strides[1] == 2 && m_strides[2] == 1)
  {
    return apply_s<2, 1>(A, kernel);
  }
  else if (m_strides[1] == 2 && m_strides[2] == 2 && m_groups == 1)
  {
    return apply_s<2, 2>(A, kernel);
    ;
  }
  else
  {
    std::cerr << "[ERROR] stride = (" << m_strides[1] << ", " << m_strides[2] << ")" << std::endl;
    assert(false);
    exit(-1);
  }
  return false;
}

template<typename T> template<int s_h, int s_w> bool Conv2D<T>::apply_s(const Tensor<T> &A, const Tensor<T> &kernel)
{
  int       in_H{ A.dims()[1] };
  int       in_W{ A.dims()[2] };
  const int half_size_h{ kernel.dims()[0] / 2 };
  const int half_size_w{ kernel.dims()[1] / 2 };
  const int k_size_h{ kernel.dims()[0] };
  const int k_size_w{ kernel.dims()[1] };

  const int top{ m_pads[0] };
  const int left{ m_pads[1] };
  int       start_h{ half_size_h - top };
  int       start_w{ half_size_w - left };
  assert(in_H > 1);
  assert(in_W > 1);

  if ((half_size_h == 2 && top != 2) || (half_size_w == 2 && left != 2))
  {
    std::cerr << "[ERROR] pad=0 not implemented for conv size 5." << std::endl;
  }
  else if ((half_size_h == 3 && top != 3) || (half_size_w == 3 && left != 3))
  {
    std::cerr << "[ERROR] pad=0 not implemented for conv size 7." << std::endl;
  }
  
  if (m_groups == 1)
  {
    if (half_size_h == 0 && half_size_w == 0)   // 1x1
    {
      conv2d_1x1_s_dispatch<s_h, s_w>(A, kernel);
    }
    else if (k_size_h == 2 && k_size_w == 2)   // 1x1
    {
      conv2d_2x2_s<s_h, s_w>(A, kernel);
    }
    else if (half_size_h == 1 && half_size_w == 1)   // 3x3
    {
      if (top == 0 && left == 0 && s_h == 1 && s_w == 1) // top=0 and left=0 means pad=(0,0)
      {
        // Do not use modified slow function -- too slow.
        //conv2d(A, kernel);

        // In order to use the fast functions, three things need to be done:
        //   I) Resize the destination matrix so that it is as big as if no padding were used ('same').
        //  II) Trick the fast function into thinking it is acutally using padding=(1,1)
        // III) Copy data back from the larger destination matrix into the correct, smaller size.

        // Create output tensor the same size as the current tensor
        Tensor<T> T2(m_out.dims());
        // Use same quantizer and dimensions
        T2.quantizer = m_out.quantizer;
        T2.border_skip = m_out.border_skip;

        //   I) Resize the destination matrix so that it is as big as if no padding were used ('same').
        Dimensions d;
        d.resize(4);
        d[0] = m_out.dims()[0];
        d[1] = m_out.dims()[1]+2;
        d[2] = m_out.dims()[2]+2;
        d[3] = m_out.dims()[3];
        m_out.resize(d);

        //  II) Trick the fast function into thinking it is acutally using padding=(1,1).
        m_pads[0] = 1;
        m_pads[1] = 1;
        conv2d_3x3_s_core_dispatch<s_h, s_w>(A, kernel);
        // Restoring m_pads so they are correct henceforth.
        m_pads[0] = top;
        m_pads[1] = left;
        
        // III) Copy data back from the larger destination matrix into the correct, smaller size.
        for(int batch = 0; batch < T2.dims()[0]; batch++)
          for(int yy = 0; yy < T2.dims()[1]; yy++)
            for(int xx = 0; xx < T2.dims()[2]; xx++)
              for(int ch = 0; ch < T2.dims()[3]; ch++)
                T2(batch,yy,xx,ch) = m_out(batch,yy+1,xx+1,ch);
        swap(m_out, T2);
      }
      else
      {
        if (!Tensor<T>::skip_border)
        {
          conv2d_3x3_s_peel<s_h, s_w>(A, kernel);
        }
        else
        {   // skip border
          if (s_h == 1 && s_w == 1)
          {
            start_h += m_out.border_skip.first;
            start_w += m_out.border_skip.second;
            in_H -= m_out.border_skip.first;
            in_W -= m_out.border_skip.second;
            m_out.border_skip.first++;
            m_out.border_skip.second++;
          }
        }
        conv2d_3x3_s_core_dispatch<s_h, s_w>(A, kernel);
      }
    }
    else if (half_size_h == 2 && half_size_w == 2)   // 5x5
    {
      if (Tensor<T>::skip_border)
      {
        std::cerr << "[ERROR] skip border with 5x5 not supported" << std::endl;
        assert(false);
        exit(-1);
      }
      conv2d_5x5_s<s_h, s_w>(A, kernel);
    }
    else if (half_size_h == 0 && half_size_w == 1 && top == 0 && left == 0 && s_h == 1 && s_w == 1)   // 1x3 pad (0,0)
    {
        // First make the output tensor larger to accommodate an output of size (n+2)x(n+2) so that
        // we can use the fast version of conv2d that is built for that.

        // Create output tensor the same size as the current tensor
        Tensor<T> T2(m_out.dims());
        // Use same quantizer and dimensions
        T2.quantizer = m_out.quantizer;
        T2.border_skip = m_out.border_skip;

        // Now change the size of the current tensor to be a bit bigger
        Dimensions d;
        d.resize(4);
        d[0] = m_out.dims()[0];
        d[1] = m_out.dims()[1];
        d[2] = m_out.dims()[2]+2;
        d[3] = m_out.dims()[3];
        m_out.resize(d);

        // Use the fast version
        m_pads[0] = 0;
        m_pads[1] = 1;
        //conv2d_ixj_s_peel<s_h, s_w>(A, kernel);
        conv2d_ixj_s_core_dispatch<s_h, s_w>(A, kernel);
        // Restoring m_pads so they are correct henceforth.
        m_pads[0] = top;
        m_pads[1] = left;

        // Copy back the lower right part to the smaller tensor
        for(int batch = 0; batch < T2.dims()[0]; batch++)
          for(int yy = 0; yy < T2.dims()[1]; yy++)
            for(int xx = 0; xx < T2.dims()[2]; xx++)
              for(int ch = 0; ch < T2.dims()[3]; ch++)
                T2(batch,yy,xx,ch) = m_out(batch,yy,xx+1,ch);
        swap(m_out, T2);
    }
    else if (half_size_h == 1 && half_size_w == 0 && top == 0 && left == 0)   // 3x1 pad (0,0)
    {
        // assert 3x3, assert stride = 1, assert pad=0
        assert(s_h == 1);
        assert(s_w == 1);
      
        // First make the output tensor larger to accommodate an output of size (n+2)x(n+2) so that
        // we can use the fast version of conv2d that is built for that.

        // Create output tensor the same size as the current tensor
        Tensor<T> T2(m_out.dims());
        // Use same quantizer and dimensions
        T2.quantizer = m_out.quantizer;
        T2.border_skip = m_out.border_skip;

        // Now change the size of the current tensor to be a bit bigger
        Dimensions d;
        d.resize(4);
        d[0] = m_out.dims()[0];
        d[1] = m_out.dims()[1]+2;
        d[2] = m_out.dims()[2];
        d[3] = m_out.dims()[3];
        m_out.resize(d);
        
        // Use the fast version
        m_pads[0] = 1;
        m_pads[1] = 0;

        //conv2d_ixj_s_peel<s_h, s_w>(A, kernel);
        conv2d_ixj_s_core_dispatch<s_h, s_w>(A, kernel);
        m_pads[0] = top;
        m_pads[1] = left;

        // Copy back the lower right part to the smaller tensor
        for(int batch = 0; batch < T2.dims()[0]; batch++)
          for(int yy = 0; yy < T2.dims()[1]; yy++)
            for(int xx = 0; xx < T2.dims()[2]; xx++)
              for(int ch = 0; ch < T2.dims()[3]; ch++)
                T2(batch,yy,xx,ch) = m_out(batch,yy+1,xx,ch);
        swap(m_out, T2);
    }
    else if (half_size_h != half_size_w)   // ixj
    {
      if (!Tensor<T>::skip_border)
      {
        conv2d_ixj_s_peel<s_h, s_w>(A, kernel);
      }
      else
      {   // skip border
        std::cerr << "[ERROR] skip border with ixj not supported" << std::endl;
        assert(false);
        exit(-1);
      }
      conv2d_ixj_s_core_dispatch<s_h, s_w>(A, kernel);
    }
    else
    {
      assert(false);
      // conv2d()
      return false;
    }
  }
  else   // groups
  {
    if (half_size_h == 1 && half_size_w == 1 && top == 0 && left == 0)   // 3x3 padding = (0,0)
    {
      // assert 3x3, assert stride = 1, assert pad=0
      assert(s_h == 1);
      assert(s_w == 1);
      // Use modified slow function
      conv2d(A, kernel);
    }
    else if (half_size_h == 0 && half_size_w == 1 && top == 0 && left == 0)   // 1x3 padding = (0,0)
    {
      // assert stride = 1, assert pad=0
      assert(s_h == 1);
      assert(s_w == 1);
    
      // First make the output tensor larger to accommodate an output of size (n)x(n+2) so that
      // we can use the fast version of conv2d that is built for that.

      // Create output tensor the same size as the current tensor
      Tensor<T> T2(m_out.dims());
      // Use same quantizer and dimensions
      T2.quantizer = m_out.quantizer;
      T2.border_skip = m_out.border_skip;

      // Now change the size of the current tensor to be a bit bigger
      Dimensions d;
      d.resize(4);
      d[0] = m_out.dims()[0];
      d[1] = m_out.dims()[1];
      d[2] = m_out.dims()[2]+2;
      d[3] = m_out.dims()[3];
      m_out.resize(d);
      
      // Use the fast version
      conv2d_ixj_s_peel<s_h, s_w>(A, kernel);
      conv2d_ixj_s_core_dispatch<s_h, s_w>(A, kernel);

      // Copy back the lower right part to the smaller tensor
      for(int batch = 0; batch < T2.dims()[0]; batch++)
        for(int yy = 0; yy < T2.dims()[1]; yy++)
          for(int xx = 0; xx < T2.dims()[2]; xx++)
            for(int ch = 0; ch < T2.dims()[3]; ch++)
              T2(batch,yy,xx,ch) = m_out(batch,yy,xx+1,ch);
      swap(m_out, T2);
    }
    else if (half_size_h == 1 && half_size_w == 0 && top == 0 && left == 0)   // 3x1 padding = (0,0)
    {
      // assert stride = 1, assert pad=0
      assert(s_h == 1);
      assert(s_w == 1);
      
      // First make the output tensor larger to accommodate an output of size (n+2)x(n) so that
      // we can use the fast version of conv2d that is built for that.

      // Create output tensor the same size as the current tensor
      Tensor<T> T2(m_out.dims());
      // Use same quantizer and dimensions
      T2.quantizer = m_out.quantizer;
      T2.border_skip = m_out.border_skip;

      // Now change the size of the current tensor to be a bit bigger
      Dimensions d;
      d.resize(4);
      d[0] = m_out.dims()[0];
      d[1] = m_out.dims()[1]+2;
      d[2] = m_out.dims()[2];
      d[3] = m_out.dims()[3];
      m_out.resize(d);
      
      // Use the fast version
      conv2d_ixj_s_peel<s_h, s_w>(A, kernel);
      conv2d_ixj_s_core_dispatch<s_h, s_w>(A, kernel);

      // Copy back the lower right part to the smaller tensor
      for(int batch = 0; batch < T2.dims()[0]; batch++)
        for(int yy = 0; yy < T2.dims()[1]; yy++)
          for(int xx = 0; xx < T2.dims()[2]; xx++)
            for(int ch = 0; ch < T2.dims()[3]; ch++)
              T2(batch,yy,xx,ch) = m_out(batch,yy+1,xx,ch);
      swap(m_out, T2);
    }
    else if (half_size_h != half_size_w)   // ixj
    {
      if (!Tensor<T>::skip_border)
      {
        conv2d_ixj_s_peel<s_h, s_w>(A, kernel);
      }
      else
      {   // skip border
        std::cerr << "[ERROR] skip border with ixj not supported" << std::endl;
        assert(false);
        exit(-1);
      }
      conv2d_ixj_s_core_dispatch<s_h, s_w>(A, kernel);
    }
    else
    {
      conv2d(A, kernel);
    }
  }
  return true;
}

// data [batch, in_height, in_width, in_channels]
// kernel [filter_height, filter_width, in_channels, out_channels]
template<typename T> bool Conv2D<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  SADL_DBG(std::cout << "  - input conv2d: " << in[0]->dims() << ' ' << in[1]->dims() << std::endl);
  if (in[0]->dims().size() != 4)
    return false;
  if (in[1]->dims().size() != 4)
    return false;
  //if ((in[1]->dims()[0]) % 2 == 0)   // even kernel
  //  return false;
  if (in[0]->dims()[0] != 1)
    return false;
  if (in[0]->dims()[0] != 1)
    return false;
  // more generic: test if padding == same
  const int in_H{ in[0]->dims()[1] };
  const int in_W{ in[0]->dims()[2] };
  const int k_h{ in[1]->dims()[0] };
  const int k_w{ in[1]->dims()[1] };
  const int s_h   = m_strides[1];
  const int s_w   = m_strides[2];
  const int out_H = (int)floor((float) (in_H + m_pads[0] + m_pads[2] - k_h) / s_h + 1);
  const int out_W = (int)floor((float) (in_W + m_pads[1] + m_pads[3] - k_w) / s_w + 1);

  // Hout=floor((H+2*p-k)/s+1)
  // assume p =k/2 (pad == same)
  // and kernel even  => Hout=ceil(H/s)
  Dimensions dim;
  dim.resize(4);
  dim[0] = in[0]->dims()[0];
  //dim[1] = (int) ceil(in[0]->dims()[1] / (float) m_strides[1]);
  //dim[2] = (int) ceil(in[0]->dims()[2] / (float) m_strides[2]);
  dim[1] = out_H;
  dim[2] = out_W;
  dim[3] = in[1]->dims()[2];
  if (out_H != dim[1] || out_W != dim[2])   // warning, fail with tf2 3x3s2 pad=same i=(5,4)
    return false;
  m_out.resize(dim);
  if (m_groups != 1)
  {
    static bool once = true;
    if (once)
      std::cout << "[WARNING] generic support for groups !=1 only for debug, do not use" << std::endl;
    once = false;
  }
  SADL_DBG(std::cout << "  - output Conv2D: " << m_out.dims() << std::endl);
  m_initDone = true;
  return true;
}

template<typename T> bool Conv2D<T>::loadInternal(std::istream &file, Version v)
{
  int32_t x = 0;
  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
    return false;
  }
  m_strides.resize(x);
  for (int k = 0; k < m_strides.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    m_strides[k] = x;
  }
  if (m_strides.size() == 2)
  {
    m_strides = Dimensions({ 1, m_strides[0], m_strides[1], 1 });
  }
  if (m_strides.size() != 4)
  {
    std::cerr << "[ERROR] invalid strides: " << m_strides.size() << std::endl;
    return false;
  }
  if (m_strides[0] != 1)
  {
    std::cerr << "[ERROR] invalid strides[0]: " << m_strides[0] << std::endl;
    return false;
  }
  if (m_strides[3] != 1)
  {
    std::cerr << "[ERROR] invalid strides[3]: " << m_strides[3] << std::endl;
    return false;
  }
  if (m_strides[1] != 1 && m_strides[1] != 2)
  {
    std::cerr << "[ERROR] not1 or 2: to check " << m_strides << std::endl;
    return false;
  }
  if (m_strides[2] != 1 && m_strides[2] != 2)
  {
    std::cerr << "[ERROR] not1 or 2: to check " << m_strides << std::endl;
  }
  SADL_DBG(std::cout << "  - strides: " << m_strides << std::endl);

  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
    return false;
  }
  m_pads.resize(x);
  for (int k = 0; k < m_pads.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    m_pads[k] = x;
  }
  SADL_DBG(std::cout << "  - pads: " << m_pads << std::endl);

  if ((int) v > 2)
  {
    file.read((char *) &m_groups, sizeof(m_groups));
    SADL_DBG(std::cout << "  - groups: " << m_groups << std::endl);
  }

  {
    file.read((char *) &m_q, sizeof(m_q));
    SADL_DBG(std::cout << "  - q: " << m_q << std::endl);
  }

  return true;
}

// should never be used for perf reasons
template<typename T> void Conv2D<T>::conv2d(const Tensor<T> &A, const Tensor<T> &kernel)
{
  const int in_H{ A.dims()[1] };
  const int in_W{ A.dims()[2] };
  const int in_D{ A.dims()[3] };
  const int nb_filters{ kernel.dims()[2] };
  const int half_size_i{ kernel.dims()[0] / 2 };
  const int half_size_j{ kernel.dims()[1] / 2 };
  const int top{ m_pads[0] };
  const int left{ m_pads[1] };
  const int start_h{ half_size_i - top };
  const int start_w{ half_size_j - left };
  const int stop_h{ in_H - start_h};
  const int stop_w{ in_W - start_w};
  const int s_h = m_strides[1];
  const int s_w = m_strides[2];
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] debug generic version conv inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W
            << " groups=" << m_groups << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC"
            << std::endl;
#endif
  constexpr int im_nb     = 0;
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < stop_h; im_i += s_h)
    {
      for (int im_j = start_w; im_j < stop_w; im_j += s_w)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = -half_size_i; filter_i <= half_size_i; ++filter_i)
        {
          // fixed
          for (int filter_j = -half_size_j; filter_j <= half_size_j; ++filter_j)
          {
            // fixed
            for (int filter_d = 0; filter_d < in_D / m_groups; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = half_size_i + filter_i;
              int kj = half_size_j + filter_j;
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
        m_out(im_nb, (im_i-start_h) / s_h, (im_j-start_w) / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}


template<typename T> template<int n> void Conv2D<T>::multiply_add_n_points(const T* input, const T coeff, typename ComputationType<T>::type* sum)
{
  for (int i = 0; i < n; ++i)
  {
    sum[i] += input[i] * coeff;
  }
}

#if __AVX2__
template<> inline void Conv2D<int16_t>::simd_multiply_add_16_points(const int16_t* input, const int16_t coeff, typename ComputationType<int16_t>::type* sum)
{
  __m256i coeff1 = _mm256_set1_epi16(coeff);
  __m256i data = _mm256_loadu_si256((__m256i*)input);
  __m256i hi = _mm256_mulhi_epi16(coeff1, data);
  __m256i lo = _mm256_mullo_epi16(coeff1, data);
  __m256i tmp1 = _mm256_unpacklo_epi16(lo, hi);
  __m256i tmp2 = _mm256_unpackhi_epi16(lo, hi);
  __m256i sum1 = _mm256_permute2x128_si256(tmp1, tmp2, 0x20);
  __m256i sum2 = _mm256_permute2x128_si256(tmp1, tmp2, 0x31);
  sum1 = _mm256_add_epi32(sum1, _mm256_loadu_si256((__m256i*)(sum + 0)));
  sum2 = _mm256_add_epi32(sum2, _mm256_loadu_si256((__m256i*)(sum + 8)));
  _mm256_storeu_si256((__m256i*)(sum + 0), sum1);
  _mm256_storeu_si256((__m256i*)(sum + 8), sum2);
}
#endif

template<typename T> inline void Conv2D<T>::simd_multiply_add_16_points(const T* input, const T coeff, typename ComputationType<T>::type* sum)
{
  for (int i = 0; i < 16; ++i)
  {
    sum[i] += input[i] * coeff;
  }
}

template<typename T> template<int s_h, int s_w> bool Conv2D<T>::conv2d_core(const Tensor<T> &A, const Tensor<T> &kernel)
{
  if constexpr (s_w != 1) // data is not continous for SIMD
    return false;
  
  constexpr int im_nb = 0;
  const int     ihalf_size{ kernel.dims()[0] / 2 };
  const int     jhalf_size{ kernel.dims()[1] / 2 };
  const int     in_D{ A.dims()[3] };
  const int     nb_filters{ kernel.dims()[2] };
  const int     shift     = kernel.quantizer + m_q;
  const int     cout_by_g = nb_filters / m_groups;
  const int     cin_by_g  = in_D / m_groups;
  const int     top{ m_pads[0] };
  const int     left{ m_pads[1] };
  int           in_H{ A.dims()[1] };
  int           in_W{ A.dims()[2] };
  int           start_h{ ihalf_size - top };
  int           start_w{ jhalf_size - left };
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv2d_core (stride known) " << kernel.dims()[0] << "x" << kernel.dims()[1] << "g" << m_groups << " inD=" << in_D << " outD=" << nb_filters
            << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << "?? kMAC" << std::endl;
#endif
  
  // TODO: SIMD version for kernel size larger than 3
  if (ihalf_size > 1 || jhalf_size > 1)
  {
    return false;
  }

  assert(start_h + s_h - ihalf_size >= 0);
  assert(start_w + s_w - jhalf_size >= 0);
  
  const int maxD = in_D / m_groups;

  if (maxD > dSize || in_H > bufSize || in_W > bufSize
    || (ihalf_size == 0 && jhalf_size == 0)) // when both are 0, SIMD may be achieved along in_D
  {
    // unsupported cases
    return false;
  }

  // for the case the block width is not vector(simd)-size aligned
  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    // group source by filter
    for (int filter_d = 0; filter_d < maxD; ++filter_d)
    {
      for (int im_i = start_h; im_i < in_H; ++im_i)
      {
        for (int im_j = start_w; im_j < in_W; ++im_j)
        {
          input_tempo[filter_d][im_i][im_j] = A(im_nb, im_i, im_j, offset + filter_d);
        }
      }
    }
    // convolution
    const bool startWithWH = ihalf_size > 0 || jhalf_size > 0;
    const int start_h2 = start_h + (startWithWH ? s_h : 0);
    const int start_w2 = start_w + (startWithWH ? s_w : 0);
    int ioffset_if_1D = ihalf_size==0 ? -1 : 0;
    int joffset_if_1D = jhalf_size==0 ? -1 : 0;
    for (int im_i = start_h2 + ioffset_if_1D; im_i < in_H - ihalf_size; im_i += s_h)
    {
      for (int im_jv = start_w2 + joffset_if_1D; im_jv < in_W - jhalf_size; im_jv += vectorSize)
      {
        typename ComputationType<T>::type x[vectorSize] = {0};
        const int max_j = std::min(im_jv + vectorSize, in_W - jhalf_size);
        for (int filter_d = 0; filter_d < maxD; ++filter_d)
        {
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {
            int ii = im_i + filter_i;
            int ki = ihalf_size + filter_i;
            for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
            {
              const int kj = jhalf_size + filter_j;
#if __AVX2__ 
              if constexpr (std::is_same_v<T, int16_t> && vectorSize == 16)
              {
                simd_multiply_add_16_points(&input_tempo[filter_d][ii][im_jv + filter_j], kernel(ki, kj, filter, filter_d), &x[0]);
              }
              else
#endif
              {
                multiply_add_n_points<vectorSize>(&input_tempo[filter_d][ii][im_jv + filter_j], kernel(ki, kj, filter, filter_d), &x[0]);
              }
              for (int im_j = im_jv; im_j < max_j; im_j += s_w)
              {
                COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
              }
            }
          }
        }      
        // only keep valid data here
        for (int im_j = im_jv; im_j < max_j; im_j += s_w)
        {
          ComputationType<T>::quantize(x[im_j - im_jv], shift);
          COUNTERS(x[im_j - im_jv]);
          SATURATE(x[im_j - im_jv]);
          m_out(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x[im_j - im_jv]);
        }
      }
    }
  }

  return true;
}


}   // namespace layers
}   // namespace sadl
#include "layer_conv2d_1x1.h"
#include "layer_conv2d_3x3.h"
#include "layer_conv2d_5x5.h"
#include "layer_conv2d_ixj.h"
#include "layer_conv2d_2x2.h"
