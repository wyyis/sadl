/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2010-2022, ITU/ISO/IEC
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
  using Layer<T>::out_;   // to avoid this->
  using Layer<T>::initDone_;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version /*v*/) override;
  Dimensions   strides_;
  Dimensions   pads_;
  int          q_      = 0;
  int          groups_ = 1;

  template<int s_h, int s_w> bool apply_s(const Tensor<T> &A, const Tensor<T> &kernel);

  // should never be used
  void conv2d(Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel);

  template<int s_h, int s_w> void conv2d_5x5_s(const Tensor<T> &A, const Tensor<T> &kernel);

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
#endif
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
  out_.quantizer          = A.quantizer - q_;
  out_.border_skip        = A.border_skip;

  assert(out_.quantizer >= 0);
  assert(kernel.quantizer + q_ >= 0);
#define DEBUG_CONV2D 0
#if DEBUG_CONV2D
  std::cout << "[WARNING] generic conv2d for debug" << std::endl;
  conv2d(out_, A, kernel);
  return true;
#endif
  if (strides_[1] == 1 && strides_[2] == 1)
  {
    return apply_s<1, 1>(A, kernel);
  }
  else if (strides_[1] == 1 && strides_[2] == 2 && groups_ == 1)
  {
    return apply_s<1, 2>(A, kernel);
  }
  else if ((strides_[1] == 2 && strides_[2] == 1) && groups_ == 1)
  {
    return apply_s<2, 1>(A, kernel);
  }
  else if (strides_[1] == 2 && strides_[2] == 2 && groups_ == 1)
  {
    return apply_s<2, 2>(A, kernel);
    ;
  }
  else
  {
    std::cerr << "[ERROR] stride = (" << strides_[1] << ", " << strides_[2] << ")" << std::endl;
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
  const int top{ pads_[0] };
  const int left{ pads_[1] };
  int       start_h{ half_size_h - top };
  int       start_w{ half_size_w - left };
  assert(in_H > 1);
  assert(in_W > 1);

  if (groups_ == 1)
  {
    if (half_size_h == 0 && half_size_w == 0)   // 1x1
    {
      conv2d_1x1_s_dispatch<s_h, s_w>(A, kernel);
    }
    else if (half_size_h == 1 && half_size_w == 1)   // 3x3
    {
      if (!Tensor<T>::skip_border)
      {
        conv2d_3x3_s_peel<s_h, s_w>(A, kernel);
      }
      else
      {   // skip border
        if (s_h == 1 && s_w == 1)
        {
          start_h += out_.border_skip;
          start_w += out_.border_skip;
          in_H -= out_.border_skip;
          in_W -= out_.border_skip;
          out_.border_skip++;
        }
      }
      conv2d_3x3_s_core_dispatch<s_h, s_w>(A, kernel);
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
    if (half_size_h != half_size_w)   // ixj
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
      conv2d(out_, A, kernel);
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
  if ((in[1]->dims()[0]) % 2 == 0)   // even kernel
    return false;
  if (in[0]->dims()[0] != 1)
    return false;
  if (in[0]->dims()[0] != 1)
    return false;
  // more generic: test if padding == same
  const int in_H{ in[0]->dims()[1] };
  const int in_W{ in[0]->dims()[2] };
  const int k_h{ in[1]->dims()[0] };
  const int k_w{ in[1]->dims()[1] };
  const int s_h   = strides_[1];
  const int s_w   = strides_[2];
  int       out_H = floor((float) (in_H + pads_[0] + pads_[2] - k_h) / s_h + 1);
  int       out_W = floor((float) (in_W + pads_[1] + pads_[3] - k_w) / s_w + 1);

  // Hout=floor((H+2*p-k)/s+1)
  // assume p =k/2 (pad == same)
  // and kernel even  => Hout=ceil(H/s)
  Dimensions dim;
  dim.resize(4);
  dim[0] = in[0]->dims()[0];
  dim[1] = (int) ceil(in[0]->dims()[1] / (float) strides_[1]);
  dim[2] = (int) ceil(in[0]->dims()[2] / (float) strides_[2]);
  dim[3] = in[1]->dims()[2];
  if (out_H != dim[1] || out_W != dim[2])   // warning, fail with tf2 3x3s2 pad=same i=(5,4)
    return false;
  out_.resize(dim);
  if (groups_ != 1)
  {
    static bool once = true;
    if (once)
      std::cout << "[WARNING] generic support for groups !=1 only for debug, do not use" << std::endl;
    once = false;
  }
  SADL_DBG(std::cout << "  - output Conv2D: " << out_.dims() << std::endl);
  initDone_ = true;
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
  strides_.resize(x);
  for (int k = 0; k < strides_.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    strides_[k] = x;
  }
  if (strides_.size() == 2)
  {
    strides_ = Dimensions({ 1, strides_[0], strides_[1], 1 });
  }
  if (strides_.size() != 4)
  {
    std::cerr << "[ERROR] invalid strides: " << strides_.size() << std::endl;
    return false;
  }
  if (strides_[0] != 1)
  {
    std::cerr << "[ERROR] invalid strides[0]: " << strides_[0] << std::endl;
    return false;
  }
  if (strides_[3] != 1)
  {
    std::cerr << "[ERROR] invalid strides[3]: " << strides_[3] << std::endl;
    return false;
  }
  if (strides_[1] != 1 && strides_[1] != 2)
  {
    std::cerr << "[ERROR] not1 or 2: to check " << strides_ << std::endl;
    return false;
  }
  if (strides_[2] != 1 && strides_[2] != 2)
  {
    std::cerr << "[ERROR] not1 or 2: to check " << strides_ << std::endl;
  }
  SADL_DBG(std::cout << "  - strides: " << strides_ << std::endl);

  file.read((char *) &x, sizeof(x));
  if (x <= 0 || x > Dimensions::MaxDim)
  {
    std::cerr << "[ERROR] invalid nb of dimensions: " << x << std::endl;
    return false;
  }
  pads_.resize(x);
  for (int k = 0; k < pads_.size(); ++k)
  {
    file.read((char *) &x, sizeof(x));
    pads_[k] = x;
  }
  SADL_DBG(std::cout << "  - pads: " << pads_ << std::endl);

  if ((int) v > 2)
  {
    file.read((char *) &groups_, sizeof(groups_));
    SADL_DBG(std::cout << "  - groups: " << groups_ << std::endl);
  }

  {
    file.read((char *) &q_, sizeof(q_));
    SADL_DBG(std::cout << "  - q: " << q_ << std::endl);
  }

  return true;
}

// should never be used for perf reasons
template<typename T> void Conv2D<T>::conv2d(Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
{
  const int in_H{ A.dims()[1] };
  const int in_W{ A.dims()[2] };
  const int in_D{ A.dims()[3] };
  const int nb_filters{ kernel.dims()[2] };
  const int half_size_i{ kernel.dims()[0] / 2 };
  const int half_size_j{ kernel.dims()[1] / 2 };
  const int top{ pads_[0] };
  const int left{ pads_[1] };
  const int start_h{ half_size_i - top };
  const int start_w{ half_size_j - left };
  const int s_h = strides_[1];
  const int s_w = strides_[2];
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] debug generic version conv inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W
            << " groups=" << groups_ << " " << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC"
            << std::endl;
#endif
  constexpr int im_nb     = 0;
  const int     shift     = kernel.quantizer + q_;
  const int     cout_by_g = nb_filters / groups_;
  const int     cin_by_g  = in_D / groups_;
  for (int filter = 0; filter < nb_filters; ++filter)
  {
    int offset = (filter / cout_by_g) * cin_by_g;
    for (int im_i = start_h; im_i < in_H; im_i += s_h)
    {
      for (int im_j = start_w; im_j < in_W; im_j += s_w)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = -half_size_i; filter_i <= half_size_i; ++filter_i)
        {
          // fixed
          for (int filter_j = -half_size_j; filter_j <= half_size_j; ++filter_j)
          {
            // fixed
            for (int filter_d = 0; filter_d < in_D / groups_; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = half_size_i + filter_i;
              int kj = half_size_j + filter_j;
              if (A.in(im_nb, ii, jj, offset + filter_d))
              {
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
              }
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_h, im_j / s_w, filter) = static_cast<T>(x);
      }
    }
  }
}

}   // namespace layers
}   // namespace sadl
#include "layer_conv2d_1x1.h"
#include "layer_conv2d_3x3.h"
#include "layer_conv2d_5x5.h"
#include "layer_conv2d_ixj.h"
