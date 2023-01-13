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
template<int s_h,int s_w>
void Conv2D<T>::conv2d_5x5_s(const Tensor<T> &A,const Tensor<T> &kernel)
{
  int       in_H{ A.dims()[1] };
  int       in_W{ A.dims()[2]  };
  const int in_D{ A.dims()[3] };
  const int nb_filters{ kernel.dims()[2] };
  constexpr int half_size{ 5 / 2 };
  const int top{ pads_[0] };
  const int left{ pads_[1] };
  int       start_h{ half_size - top -2};
  int       start_w{ half_size - left -2};
  constexpr int im_nb = 0;
  const int     shift = kernel.quantizer + q_;
#if DEBUG_SIMD && __AVX2__
  std::cout << "\n[WARN] debug generic version conv inD=" << in_D << " outD=" << nb_filters << " s=[" << s_w << ' ' << s_h << "] " << in_H << 'x' << in_W << " "
            << in_D * kernel.dims()[0] * kernel.dims()[1] * nb_filters * (in_H / s_h) * (in_W / s_w) / 1000 << " kMAC" << std::endl;
#endif

  for (int filter = 0; filter < nb_filters; ++filter)
  {
    for (int im_i = start_h + s_h; im_i < in_H  /*- s_h*/; im_i += s_h)
    {
      for (int im_j = start_w + s_w; im_j < in_W /*- s_w*/; im_j += s_w)
      {
        typename ComputationType<T>::type x = 0;
        for (int filter_i = -half_size; filter_i <= half_size; ++filter_i)
        {
          // fixed
          for (int filter_j = -half_size; filter_j <= half_size; ++filter_j)
          {
            // fixed
            for (int filter_d = 0; filter_d < in_D; ++filter_d)
            {
              int ii = im_i + filter_i;
              int jj = im_j + filter_j;
              int ki = half_size + filter_i;
              int kj = half_size + filter_j;
              if (A.in(im_nb, ii, jj, filter_d))
              {
                x += (typename ComputationType<T>::type) A(im_nb, ii, jj, filter_d) * kernel(ki, kj, filter, filter_d);
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

// NO SIMD yet
}
}


