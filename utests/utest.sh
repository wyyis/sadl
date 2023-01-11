#!/bin/bash
SPATH=$(dirname $(realpath ${BASH_SOURCE[0]}));
BDIR=$(realpath $SPATH);

if [ $# != 1 -a $# != 2 ]; then
 echo "[ERROR] utest.sh model.onnx [--no_transpose]"
 exit -1;
fi;
OPT="";
if [ $# == 2 ]; then
 OPT="$2";
fi;
 
if [ ! -f ${1} ]; then
 echo "[ERROR] no file $1"
 exit -1;
fi;

NAME=$(basename $1);
BNAME=${NAME/.onnx/};

rm -f ${BNAME}.results;
python3 ../onnx_inference.py --input_onnx $1 --output ${BNAME}.results $OPT > /dev/null
if [ ! -f ${BNAME}.results ]; then
 echo " [ERROR] onnx inference failed";
 exit -1;
fi

rm -f ${BNAME}.sadl;
python3 ../../converter/main.py --input_onnx $1  --output ${BNAME}.sadl > /dev/null
if [ ! -f ${BNAME}.sadl ]; then
 echo " [ERROR] onnx to SADL conversion failed";
 exit -1;
fi

../build/test ${BNAME}.sadl ${BNAME}.results 0.001 > /dev/null
E=$?;
if (( E == 0 )); then
  echo -e "   \e[32m[PASS]\e[0m $BNAME " 
else
  echo -e "   \e[31m[FAIL]\e[0m $BNAME"
  exit -1;  
fi;

