# how-to-optimize-gemm
RowMajor gemm optimization

原文为**列主序**`x86 SSE`代码。
考虑到移动端卷积优化一般使用`arm`架构芯片，
因此基于 [blis-lab](https://github.com/flame/blislab) 文档和[项目](https://github.com/flame/how-to-optimize-gemm)实现`arm64`版**行主序**`gemm`优化。目前最新的`MMult_4x4_17.c`最高可达 9.9gflops，相当于 CPU 峰值的 70%。

本系列优化中文教程在

1. [知乎 GEMM 入门](https://zhuanlan.zhihu.com/p/65436463)
2. [知乎 GEMM caching](https://zhuanlan.zhihu.com/p/69700540)
