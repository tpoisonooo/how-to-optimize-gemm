#!/bin/bash
set -e

VERSIONS=(
  MMult_cuBLAS_1
  MMult_cuBLAS_2
  MMult_cuda_2
  MMult_cuda_3
  MMult_cuda_4
  MMult_cuda_5
  MMult_cuda_6
  MMult_cuda_7
  MMult_cuda_8
  MMult_cuda_9
  MMult_cuda_10
  MMult_cuda_11
  MMult_cuda_12
)

for v in "${VERSIONS[@]}"; do
  echo "========================================"
  echo "Testing $v ..."
  echo "========================================"
  make clean >/dev/null 2>&1
  if ! make NEW=$v SMS="70 75 80 86 90" run > /tmp/${v}_log.txt 2>&1; then
    echo "FAILED: $v (build/run error)"
    cat /tmp/${v}_log.txt
    exit 1
  fi
  
  if grep -q "diff too big" /tmp/${v}_log.txt; then
    echo "FAILED: $v (diff too big)"
    cat /tmp/${v}_log.txt
    exit 1
  fi
  
  if grep -q "error:" /tmp/${v}_log.txt; then
    echo "FAILED: $v (compilation error)"
    cat /tmp/${v}_log.txt
    exit 1
  fi
  
  echo "PASSED: $v"
  tail -n 2 output_$v.m
  echo ""
done

echo "========================================"
echo "All versions passed!"
echo "========================================"
