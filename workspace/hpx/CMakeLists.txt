# HPX examples: an algorithm with the sequential policy, asynchronous tasks and a parallel algorithm on
# the CPU, and CUDA kernels driven by an HPX executor on the GPU.
cmake_minimum_required(VERSION 3.25)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)  # the GPU of this instance
endif()
project(hpx_examples LANGUAGES CXX CUDA)

find_package(HPX REQUIRED)

# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-march=x86-64-v3>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=x86-64-v3>")

add_executable(pi_seq pi_seq.cpp)
add_executable(pi_cpu pi_cpu.cpp)
add_executable(pi_gpu pi_gpu.cu)
add_executable(matmul_seq matmul_seq.cpp)
add_executable(matmul_cpu matmul_cpu.cpp)
add_executable(matmul_gpu matmul_gpu.cu)
foreach(example pi_seq pi_cpu pi_gpu matmul_seq matmul_cpu matmul_gpu)
  target_link_libraries(${example} HPX::hpx HPX::wrap_main)
endforeach()
