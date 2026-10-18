# Matrix product C = A B on the GPU with CUDA.jl array programming, without Dagger: A * B on CuArrays
# calls cuBLAS (SGEMM), a tuned library instead of the simple algorithm of the other versions. The time
# includes copying A and B to the GPU and C back into a host array.
using CUDA
include("common.jl")

matmul_gpu!(C, A, B) = copyto!(C, CuArray(A) * CuArray(B))

CUDA.functional() || error("CUDA.jl cannot use the GPU")
A, B = matrix_init(MATRIX_N)
C = zeros(Float32, size(A))  # written once now, so that its page faults do not fall into the timed run
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_gpu!(similar(A), A, B))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed matmul_gpu!(C, A, B)
matrix_report("GPU (CUDA.jl arrays: A * B with cuBLAS, $(CUDA.name(CUDA.device())))", A, B, C, seconds) || exit(1)
