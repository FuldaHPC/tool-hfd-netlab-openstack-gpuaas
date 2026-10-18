# Matrix product C = A B on the GPU with a KernelAbstractions.jl kernel, without Dagger: one work-item
# per element of C in workgroups of 16 x 16 (as the @cuda kernel; with KernelAbstractions' default the
# product took about 1.6 times as long), on CUDABackend() here. The time includes copying A and B to the
# GPU and C back.
using CUDA, KernelAbstractions
include("common.jl")

@kernel function matmul_kernel!(C, A, B)
    i, j = @index(Global, NTuple)
    @inbounds C[i, j] = matmul_entry(A, B, i, j)
end

function matmul_gpu!(C, A, B)
    backend = CUDABackend()
    dA, dB = CuArray(A), CuArray(B)
    dC = similar(dA)
    matmul_kernel!(backend, (16, 16))(dC, dA, dB; ndrange=size(dC))
    return copyto!(C, dC)
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
A, B = matrix_init(MATRIX_N)
C = zeros(Float32, size(A))  # written once now, so that its page faults do not fall into the timed run
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_gpu!(similar(A), A, B))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed matmul_gpu!(C, A, B)
matrix_report("GPU (KernelAbstractions, $(CUDA.name(CUDA.device())))", A, B, C, seconds) || exit(1)
