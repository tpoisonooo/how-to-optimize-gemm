#! /bin/bash

NAME=${1}
KOMPUTE_BUILD="/home/khj/kompute/build"

g++ -g -O0 -std=c++17 -c ${NAME}.cpp
g++ -o ${NAME} ${NAME}.o  "${KOMPUTE_BUILD}/src/libkompute.a" "${KOMPUTE_BUILD}/src/kompute_fmt/libfmt.a" "${KOMPUTE_BUILD}/src/kompute_spdlog/libspdlog.a" -L/usr/local/lib -lvulkan -lpthread
