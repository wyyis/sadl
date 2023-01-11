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

../utest.sh ../models/conv2d_grouped.onnx --no_transpose
../utest.sh ../models/slice.onnx --no_transpose
../utest.sh ../models/conv2d_3x3s1_10x10x3.onnx 



