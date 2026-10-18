# Kokkos examples. Kokkos was built with CMake's CUDA language support, so every source that
# includes Kokkos is compiled as CUDA; the GPU architecture comes from the Kokkos installation.
cmake_minimum_required(VERSION 3.25)
project(kokkos_examples LANGUAGES CXX CUDA)

find_package(Kokkos REQUIRED)

# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=x86-64-v3>")

foreach(example pi_seq pi_cpu pi_gpu matmul_seq matmul_cpu matmul_gpu)
  add_executable(${example} ${example}.cpp)
  set_source_files_properties(${example}.cpp PROPERTIES LANGUAGE CUDA)
  target_link_libraries(${example} Kokkos::kokkos)
endforeach()
