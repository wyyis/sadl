#!/bin/bash


echo "[INFO] BUILD SADL SAMPLE"
# build sample
mkdir -p sample_test;
cd sample_test;
cmake -DCMAKE_BUILD_TYPE=Release ../sample
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m cmake" 
else
  echo -e "   \e[31m[FAIL]\e[0m cmake"
  exit -1;  
fi;
make -j
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m build" 
else
  echo -e "   \e[31m[FAIL]\e[0m build"
  exit -1;  
fi;
echo ""

echo "[INFO] TF2 -> ONNX -> SADL"
# TF2
python3 ../sample/tf2.py 2>/dev/null # output a tf2.onnx
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m tf -> onnx" 
else
  echo -e "   \e[31m[FAIL]\e[0m tf -> onnx"
  exit -1;  
fi;
python3 ../converter/main.py --input_onnx tf2.onnx --output tf2.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m onnx -> sadl" 
else
  echo -e "   \e[31m[FAIL]\e[0m onnx -> sadl"
  exit -1;  
fi;
./sample_simd512 tf2.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl -> inference" 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl -> inference"
  exit -1;  
fi;
echo ""

echo "[INFO] PYTORCH -> ONNX -> SADL"
# torch
python3 ../sample/pytorch.py 2>/dev/null # output a pytorch.onnx
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m pytorch -> onnx" 
else
  echo -e "   \e[31m[FAIL]\e[0m pytorch -> onnx"
  exit -1;  
fi;
python3 ../converter/main.py --input_onnx pytorch.onnx --output pytorch.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m onnx -> sadl" 
else
  echo -e "   \e[31m[FAIL]\e[0m onnx -> sadl"
  exit -1;  
fi;
./sample_simd512 pytorch.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl -> inference" 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl -> inference"
  exit -1;  
fi;
echo ""

echo "[INFO] DEBUG MODEL"
./debug_model pytorch.sadl > debug_model.log
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m debug model" 
else
  echo -e "   \e[31m[FAIL]\e[0m debug model"
  exit -1;  
fi;
echo "see debug_model.log"
echo ""

echo "[INFO] COUNT MAC"
./count_mac pytorch.sadl
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m count mac" 
else
  echo -e "   \e[31m[FAIL]\e[0m count mac"
  exit -1;  
fi;
echo "[INFO] WRITE INT16 MODEL"
echo  "0 15    1 8 2 0 3 8   6 8 7 0 8 8    10 9 11 0 12 8   14 8 15 1 16 8    20 8 21 0 22 9   24 8 25 0 26 8     28 8 29 0 30 8  34 8 35 0 36 8  38 0 39 0 40 0 42 8  43 0 44 8" | ./naive_quantization pytorch.sadl pytorch_int16.sadl;
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m integerization" 
else
  echo -e "   \e[31m[FAIL]\e[0m integerization"
  exit -1;  
fi;
echo "[INFO] QUANTIZATION CONSISTENCY"
# Usage: quantization_test model_float.sadl model_int16.sadl inputs_range shift max_error
python3 ../converter/main.py --input_onnx ../utests/models/quantization_test.onnx --output quantization_test.sadl
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m onnx -> sadl" 
else
  echo -e "   \e[31m[FAIL]\e[0m onnx -> sadl"
  exit -1;  
fi;
echo "0 10    1 10 2 0 3 10" | ./naive_quantization quantization_test.sadl quantization_test_int16.sadl;
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl float -> integerization" 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl float -> integerization"
  exit -1;  
fi;
./quantization_test quantization_test.sadl quantization_test_int16.sadl 1024.0 0 3
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl int -> inference + test " 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl int -> inference + test"
  exit -1;  
fi;
echo "0 10    1 9 2 0 3 9" | ./naive_quantization quantization_test.sadl quantization_test_int16.sadl;
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl float -> integerization" 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl float -> integerization"
  exit -1;  
fi;
./quantization_test quantization_test.sadl quantization_test_int16.sadl 1024.0 1 6
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl int -> inference + test " 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl int -> inference + test"
  exit -1;  
fi;
echo ""

echo "[INFO] BUILD SADL SAMPLE SPARSE"
# build sample
mkdir -p sparse;
cd sparse;
cmake -DCMAKE_BUILD_TYPE=Release -DSPARSE_MATMULT_SUPPORT=1 ../../sample
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m cmake" 
else
  echo -e "   \e[31m[FAIL]\e[0m cmake"
  exit -1;  
fi;
make sample_simd512 debug_model
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m build" 
else
  echo -e "   \e[31m[FAIL]\e[0m build"
  exit -1;  
fi;
echo ""

echo "[INFO] PYTORCH -> ONNX -> SADL"
# torch
python3 ../../sample/pytorch_matmult.py 2>/dev/null # output a pytorch.onnx
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m pytorch -> onnx" 
else
  echo -e "   \e[31m[FAIL]\e[0m pytorch -> onnx"
  exit -1;  
fi;
python3 ../../converter/main.py --input_onnx pytorch_matmult.onnx --output pytorch_matmult.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m onnx -> sadl" 
else
  echo -e "   \e[31m[FAIL]\e[0m onnx -> sadl"
  exit -1;  
fi;
echo ""

echo "[INFO] CHECK sparsity"
./debug_model pytorch_matmult.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m debug model sparse" 
else
  echo -e "   \e[31m[FAIL]\e[0m debug model sparse"
  exit -1;  
fi;
echo ""

echo "[INFO] INFERENCE not sparse"
../sample_simd512 pytorch_matmult.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl -> inference non sparse" 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl -> inference non sparse"
  exit -1;  
fi;
echo ""

echo "[INFO] INFERENCE sparse"
./sample_simd512 pytorch_matmult.sadl 
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m sadl -> inference sparse" 
else
  echo -e "   \e[31m[FAIL]\e[0m sadl -> inference sparse"
  exit -1;  
fi;
echo ""
cd ..;


if [ -f tf2.sadl -a -f pytorch.sadl \
     -a -f sample_generic -a -f sample_simd256 -a -f sample_simd512 \
     -a -f count_mac \
     -a -f debug_model \
     -a -f naive_quantization \
     -a -f quantization_test \
     -a -f sparse/pytorch_matmult.sadl ]; then
 echo "[INFO] all build OK"
 echo -e "\e[32m[PASS]\e[0m All tests" 
 exit 0;
else
 exit 1;
fi;
