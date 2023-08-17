cmake_minimum_required(VERSION 3.15.2)
    
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")

project(how-to-optimize-gemm LANGUAGES C CXX ASM VERSION 0.1)
set(CMAKE_CXX_STANDARD 17)

option(MPERF_ENABLE "build with mperf." OFF)
set(CMAKE_CXX_FLAGS "-O2 -g -march=native -ftree-vectorize ${CMAKE_CXX_FLAGS}")

if(MPERF_ENABLE)
    add_definitions(-DMPERF=1)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../mperf ${CMAKE_CURRENT_BINARY_DIR}/mperf)
endif()

function(add_bin source_file)
    get_filename_component(target_name ${source_file} NAME_WE)
    add_executable(${target_name} ${source_file} test_MMult.cpp compare_matrices.cpp random_matrix.cpp copy_matrix.cpp dclock.cpp REF_MMult.cpp print_matrix.cpp)
    
    if(MPERF_ENABLE)
        target_link_libraries(${target_name} mperf)
    endif()
endfunction()


add_bin(MMult0.cpp)
add_bin(MMult1.cpp)
add_bin(MMult_4x4_8.cpp)
add_bin(MMult_4x4_9.cpp)
add_bin(MMult_4x4_10.cpp)
add_bin(MMult_4x4_11.cpp)
add_bin(MMult_4x4_12.cpp)
add_bin(MMult_4x4_13.cpp)
add_bin(MMult_4x4_14.cpp)
add_bin(MMult_4x4_15.cpp)
add_bin(MMult_4x4_16.cpp)
add_bin(MMult_4x4_17.cpp)
add_bin(MMult_4x4_18.cpp)
add_bin(MMult_4x4_19.cpp)

