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
#include <cmath>
#include "layer.h"
#if __AVX2__
#include <immintrin.h>
#endif

namespace sadl {
namespace layers {


template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_ixj_s_peel(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                  const Tensor<T> &kernel)
{
  constexpr int im_nb      = 0;
  const int     shift      = kernel.quantizer + q_;
  const int     ihalf_size{ kernel.dims()[0] / 2 };
  const int     jhalf_size{ kernel.dims()[1] / 2 };
  const int     cout_by_g = nb_filters / groups_;
  const int     cin_by_g  = in_D / groups_;
  for (int filter_nb = 0; filter_nb < nb_filters; ++filter_nb)
  {

    int offset = (filter_nb / cout_by_g) * cin_by_g;

    // corners
    {
      int  im_i;
      int  im_j;
      auto loop_with_cond = [&, filter_nb, shift](int i0, int i1, int j0, int j1)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = i0; filter_i <= i1; ++filter_i)
        {
          for (int filter_j = j0; filter_j <= j1; ++filter_j)
          {
            for (int filter_d = 0; filter_d < in_D/groups_; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = ihalf_size + filter_i;
              int kj = jhalf_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
            }
          }
        }
        ComputationType<T>::quantize(x, shift);
        COUNTERS(x);
        SATURATE(x);
        out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
      };

      im_j = start_w;
      if (im_j < in_W)
      {   // left side
        im_i = start_h;
        if (im_i < in_H)
        {   // top left corner
          loop_with_cond(-start_h, ihalf_size, -start_w, jhalf_size);
        }
        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom left corner
          const int end_i = (im_i + 1 < in_H) ? 1 : 0;
          loop_with_cond(-ihalf_size, end_i, -start_w, jhalf_size);
        }
      }

      im_j            = ((in_W - jhalf_size - start_w) / s_w) * s_w + start_w;
      const int end_j = (im_j + 1 < in_W) ? 1 : 0;
      if (im_j > 0 && im_j < in_W && im_j != start_w)
      {   // rihgt side
        im_i = start_h;
        if (im_i < in_H)
        {   // top right corner
          loop_with_cond(-start_h, ihalf_size, -jhalf_size, end_j);
        }

        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom right corner
          const int end_i = (im_i + 1 < in_H) ? 1 : 0;
          loop_with_cond(-ihalf_size, end_i, -jhalf_size, end_j);
        }
      }
    }

    // vertical borders
    {
      for (int im_i = start_h + s_h; im_i < in_H - ihalf_size; im_i += s_h)
      {
        int im_j = start_w;   // can be only 0 or 1
        if (im_j < in_W)
        {   // left side
          typename ComputationType<T>::type x = 0;
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -start_w; filter_j <= jhalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D/groups_; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }

        im_j = ((in_W - jhalf_size - start_w) / s_w) * s_w + start_w;
        if (im_j > 0 && im_j < in_W && im_j != start_w)
        {   // rihgt side
          typename ComputationType<T>::type x          = 0;
          const int                         end_filter = (im_j + 1) < in_W ? 1 : 0;
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -jhalf_size; filter_j <= end_filter; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D/groups_; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
      }
    }
    {
      // horizontal borders
      for (int im_j = s_w + start_w; im_j < in_W - jhalf_size; im_j += s_w)
      {
        int im_i = start_h;   // 0 or 1 -> adapt filter start
        if (im_i < in_H)
        {   // top line
          typename ComputationType<T>::type x = 0;
          for (int filter_i = -start_h; filter_i <= ihalf_size; ++filter_i)
          {
            for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D/groups_; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }

          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
        im_i = ((in_H - ihalf_size - start_h) / s_h) * s_h + start_h;
        if (im_i > 0 && im_i < in_H && im_i != start_h)
        {   // bottom line
          typename ComputationType<T>::type x          = 0;
          const int                         end_filter = (im_i + 1) < in_H ? 1 : 0;
          for (int filter_i = -ihalf_size; filter_i <= end_filter; ++filter_i)
          {
            for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
            {
              for (int filter_d = 0; filter_d < in_D/groups_; ++filter_d)
              {
                int ii = im_i + filter_i;
                int jj = im_j + filter_j;
                int ki = ihalf_size + filter_i;
                int kj = jhalf_size + filter_j;
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter_nb, filter_d);
                COUNTERS_MAC(kernel(ki, kj, filter_nb, filter_d));
              }
            }
          }
          ComputationType<T>::quantize(x, shift);
          COUNTERS(x);
          SATURATE(x);
          out_(im_nb, im_i / s_h, im_j / s_w, filter_nb) = static_cast<T>(x);
        }
      }
    }
  }   // filter_nb
}


template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_ixj_s_core_dispatch(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                           const Tensor<T> &kernel)
{
//Update SIMD function once supported
#if __AVX2__
#define CONV_MOD8 conv2d_ixj_s_d_core
#else
#define CONV_MOD8 conv2d_ixj_s_d_core
#endif

  switch (in_D)
  {
  case 24: CONV_MOD8<24, s_h, s_w>(nb_filters, in_H, in_W, start_h, start_w, out_, A, kernel); break;
  default: conv2d_ixj_s_core<s_h, s_w>(nb_filters, in_H, in_W, in_D, start_h, start_w, out_, A, kernel); break;
  }
#undef CONV_MOD8
}




template<typename T>
template<int s_h, int s_w>
void Conv2D<T>::conv2d_ixj_s_core(int nb_filters, int in_H, int in_W, int in_D, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A,
                                  const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv ixj inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
  constexpr int im_nb     = 0;
  const int     ihalf_size{ kernel.dims()[0] / 2 };
  const int     jhalf_size{ kernel.dims()[1] / 2 };
  const int     shift     = kernel.quantizer + q_;
  const int     cout_by_g = nb_filters / groups_;
  const int     cin_by_g  = in_D / groups_;
  const int top{ pads_[0] };
  const int left{ pads_[1] };
  for (int im_i = start_h + top; im_i < in_H - top; im_i += s_h)
  {
    for (int im_j = start_w + left; im_j < in_W - left; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        int offset = (filter / cout_by_g) * cin_by_g;
        typename ComputationType<T>::type x = 0;
        for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
        {   // fixed
          for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
          {   // fixed
            for (int filter_d = 0; filter_d < in_D/groups_; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = ihalf_size + filter_i;
              int kj = jhalf_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
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


template<typename T>
template<int in_D, int s_h, int s_w>
void Conv2D<T>::conv2d_ixj_s_d_core(int nb_filters, int in_H, int in_W, int start_h, int start_w, Tensor<T> &out_, const Tensor<T> &A, const Tensor<T> &kernel)
{
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] generic version conv ixj inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "]  " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif
  constexpr int im_nb     = 0;
  const int     ihalf_size{ kernel.dims()[0] / 2 };
  const int     jhalf_size{ kernel.dims()[1] / 2 };
  const int     shift     = kernel.quantizer + q_;
  const int     cout_by_g = nb_filters / groups_;
  const int     cin_by_g  = in_D / groups_;
  const int top{ pads_[0] };
  const int left{ pads_[1] };
  for (int im_i = start_h + top; im_i < in_H - top; im_i += s_h)
  {
    for (int im_j = start_w + left; im_j < in_W - left; im_j += s_w)
    {
      for (int filter = 0; filter < nb_filters; ++filter)
      {
        int offset = (filter / cout_by_g) * cin_by_g;
        typename ComputationType<T>::type x = 0;
        for (int filter_d = 0; filter_d < in_D/groups_; ++filter_d)
        {
          for (int filter_i = -ihalf_size; filter_i <= ihalf_size; ++filter_i)
          {   // fixed
            for (int filter_j = -jhalf_size; filter_j <= jhalf_size; ++filter_j)
            {   // fixed
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = ihalf_size + filter_i;
              int kj = jhalf_size + filter_j;
              x += (typename ComputationType<T>::type) A(im_nb, ii, jj, offset + filter_d) * kernel(ki, kj, filter, filter_d);
              COUNTERS_MAC(kernel(ki, kj, filter, filter_d));
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


}
}
