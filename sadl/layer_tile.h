#pragma once
#include <cstring>
#include "layer.h"

namespace sadl
{
namespace layers
{
template<typename T> class Tile : public Layer<T>
{
public:
  using Layer<T>::Layer;
  using Layer<T>::m_out;
  using Layer<T>::m_initDone;

  virtual bool apply(std::vector<Tensor<T> *> &in) override;
  virtual bool init(const std::vector<Tensor<T> *> &in) override;

protected:
  virtual bool loadInternal(std::istream &file, Version v) override;
};


template<typename T> bool Tile<T>::apply(std::vector<Tensor<T> *> &in)
{
  assert(in.size() == 2);
  m_out.quantizer = in[0]->quantizer;

  // Apply the tile
  const int in_N{ in[0]->dims()[0] };
  const int in_H{ in[0]->dims()[1] };
  const int in_W{ in[0]->dims()[2] };
  const int in_K{ in[0]->dims()[3] };

  const int out_N{ m_out.dims()[0] };
  const int out_H{ m_out.dims()[1] };
  const int out_W{ m_out.dims()[2] };
  const int out_K{ m_out.dims()[3] };

  for (int d0 = 0; d0 < out_N; ++d0) {
    for (int d1 = 0; d1 < out_H; ++d1) {
      for (int d2 = 0; d2 < out_W; ++d2) {
        int inputIdx =  d0 % in_N * (in_H * in_W * in_K) +
                        d1 % in_H * (in_W * in_K) +
                        d2 % in_W * in_K;
        int outputIdx = d0 * (out_H * out_W * out_K) +
                        d1 * (out_W * out_K) +
                        d2 * out_K;
        for (int idx = 0; idx < (*in[1])[3]; ++idx) {
            std::memcpy(&m_out[outputIdx + idx * in_K],  &((*in[0])[inputIdx]), sizeof(T) * in_K);
        }
      }
    }
  }
  
  return true;
}

template<typename T> bool Tile<T>::init(const std::vector<Tensor<T> *> &in)
{
  if (in.size() != 2)
    return false;
  if (in[0]->dims().size() != 4)
    return false;
  if (in[1]->dims().size() != 1)
    return false;
  SADL_DBG(std::cout << "input_dims: " << in[0]->dims() << " tile_dims: " << in[1]->dims() << std::endl);

  Dimensions dims;
  dims.resize((int) in[1]->size());
  if (!std::is_same<float, T>::value && in[1]->quantizer != 0) {
    std::cerr << "[ERROR] quantizer on reshape dimensions data layer" << std::endl;
    return false;
  }
  for(int i = 0; i < in[1]->size(); i++) {
    dims[i] = (int)(*in[1])[i];
  }

  if (in[1]->size() != in[0]->dims().size()) {
    std::cerr << "[ERROR] tile incompatible tile shape " << dims << " for input shape " << in[0]->dims() << std::endl;
    return false;
  }
  
  for (int i = 0; i < in[0]->dims().size(); ++i) {
    dims[i] = dims[i] * in[0]->dims()[i];
  }
  SADL_DBG(std::cerr << "output_dims: " << dims << std::endl);

  m_out.resize(dims);
  m_initDone = true;
  return true;
}

template<typename T> bool Tile<T>::loadInternal(std::istream &, Version) { return true; }

}   // namespace layers
}   // namespace sadl
