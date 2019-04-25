# how-to-optimize-gemm
RowMajor gemm optimization

考虑到移动端卷积优化一般使用`arm`架构芯片，
因此基于 [blis-lab](https://github.com/flame/blislab) 文档和[项目](https://github.com/flame/how-to-optimize-gemm)实现`arm64`版**行主序**`gemm`优化。
原文为**列主序**`x86 SSE`代码。

由于简书随意封禁文章，本系列优化中文教程临时放在[私有博客服务](http://101.201.67.56/article-detials/1，第一次打开需要加载一会儿)。
