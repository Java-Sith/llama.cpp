#!/bin/bash

#EXPERIMENTS="FIXED8 FIXED16 FLOAT4 FLOAT8 FLOAT16 FLOAT32"
EXPERIMENTS="FIXED8 FIXED16 FLOAT4 FLOAT8 FLOAT16 FLOAT32"
export BUS=2048
export B_COLS=32768
export C_COLS=32768

for i in ${EXPERIMENTS}; do
  echo "Running: ${i} experiment"
  mkdir -p solutions/experiment-${i}
  DATATYPE=${i} vitis_hls -f matmul.tcl
  cp -r matmul/solution/syn/report/* solutions/experiment-${i}
done
