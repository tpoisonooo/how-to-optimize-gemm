# how-to-optimize-gemm on cuda
cuda GEMM optimization

## 软硬件环境
自行安装 cuda 和 OpenBLAS 

## 编译和运行

1. 首次运行需要 `makefile` 里改成同一个代码版本，例如
```
OLD    := MMult_cuda_5
NEW   := MMult_cuda_5
```

```bash
$ cd src/HowToOptimizeGemm
$ make
$ ./test_MMult.x
... 
```

2. 如果想用`plot.py`绘制折线图，用 `make run` 生成运行结果


## cuda 版本

切换到  [cuda](https://github.com/tpoisonooo/how-to-optimize-gemm/tree/cuda) 分支。
目前 WIP，优化至 cuBLAS 99% 的性能后，发布知乎教程。
