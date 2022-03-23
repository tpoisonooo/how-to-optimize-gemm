# How to Build

## Fetch Vulkan SDK

```bash
$  wget https://sdk.lunarg.com/sdk/download/1.3.204.1/linux/vulkansdk-linux-x86_64-1.3.204.1.tar.gz
$ tar xvf vulkansdk-linux-x86_64-1.3.204.1.tar.gz
$ export VULKAN_SDK=/path/to/1.3.204.1/x86_64
```

## Build and Install `glslangValidator`

```bash
$ git clone git clone https://github.com/KhronosGroup/glslang.git  --recursive --depth=1
$ cd glslang
$ ./update_glslang_sources.py
$ cmake -DCMAKE_INSTALL_PREFIX="/path/to/glslang/install"  ..
$ make && make install
$ export PATH=/path/to/glslang/install/bin
```

## Build and Install `kompute`

```bash
$ git clone https://github.com/KomputeProject/kompute  --depth=1 --recursive
$ cd kompute
$ mkdir -p build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX="/path/to/kompute/install" ..
$ make && make install
```

## Build
Now we have `libkompute.a` and `glslangValidator`, edit makefile and compile our GEMM implementation.
```bash
$ vim makefile
#  update KOMPUTE_BUILD
$ export CPLUS_INCLUDE_PATH=`pwd`
$ make
...
```

## Run
On Jetson Nano, enable MAXN power mode first.

```bash
$ sudo jetson_clocks
...
$ ./test_MMult.x
...
```
