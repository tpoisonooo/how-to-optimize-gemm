# how-to-optimize-gemm
RowMajor gemm optimization

考虑到移动端卷积优化一般使用`arm`架构芯片，
因此基于 [blis-lab](https://github.com/flame/blislab) 文档和[项目](https://github.com/flame/how-to-optimize-gemm)实现`arm64`版**行主序**`gemm`优化。
原文为**列主序**`x86 SSE`代码。

本系列优化中文教程在[简书文档](https://www.jianshu.com/p/26f24f464016)
