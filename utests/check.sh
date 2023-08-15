#!/bin/bash
SPATH=$(dirname $(realpath ${BASH_SOURCE[0]}));
BDIR=$(realpath $SPATH);

cd $BDIR;
mkdir -p build;
cd build;
cmake ..;
make -j;
cd ..;

mkdir -p results;
cd results;

# no transpose
L="EE1-1.10_model0 slice scatternd_c_tf2 scatternd_hwc_with_conv_tf2 slice_hwc";
for F in $L; do
  ../utest.sh ../models/${F}.onnx --no_transpose;
done
  
L="conv2d_4_8x8x4_k1x1s1,1_g1_p0,0 conv2d_4_8x8x4_k1x1s1,1_g4_p0,0 conv2d_4_8x8x4_k1x1s2,1_g1_p0,0 conv2d_4_8x8x4_k1x3s1,2_g1_p0,1 conv2d_4_8x8x4_k3x1s1,1_g1_p1,0 conv2d_4_8x8x4_k3x1s1,1_g4_p1,0 conv2d_4_8x8x4_k3x3s1,1_g1_p1,1 conv2d_4_8x8x4_k3x3s2,1_g1_p1,1 conv2d_4_8x8x4_k3x3s2,2_g1_p1,1 conv2d_4_8x8x4_k5x5s1,1_g1_p2,2 conv2d_4_8x8x4_k5x5s1,1_g4_p2,2 conv2d_4_8x8x4_k5x5s2,1_g1_p2,2 conv2d_4_8x9x4_k1x1s2,1_g1_p0,0 conv2d_4_8x9x4_k3x1s1,1_g4_p1,0 conv2d_4_8x9x4_k3x3s1,1_g4_p1,1 conv2d_4_8x9x4_k3x3s2,1_g1_p1,1 conv2d_4_8x9x4_k3x3s2,2_g1_p1,1 conv2d_4_9x8x4_k1x1s1,1_g1_p0,0 conv2d_4_9x8x4_k1x1s2,1_g1_p0,0 conv2d_4_9x8x4_k1x3s1,2_g1_p0,1 conv2d_4_9x8x4_k3x1s1,1_g1_p1,0 conv2d_4_9x8x4_k3x3s1,1_g1_p1,1 conv2d_4_9x8x4_k3x3s2,1_g1_p1,1 conv2d_4_9x8x4_k3x3s2,2_g1_p1,1 conv2d_4_9x8x4_k5x5s1,1_g1_p2,2 conv2d_4_9x8x4_k5x5s2,1_g1_p2,2 conv2d_4_9x9x4_k1x3s1,2_g1_p0,1 repeated_conv slice_pytorch slice_inf_pytorch slice_chw_pytorch prelu_multiple_alpha prelu_single_alpha scatternd_c_pytorch scatternd_hwc_with_conv_pytorch";
for F in $L; do
  ../utest.sh ../models/${F}.onnx;
done




