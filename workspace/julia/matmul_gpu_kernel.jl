# Matrix product C = A B on the GPU with a hand-written CUDA.jl kernel (@cuda), without Dagger: one
# thread per element of C. Julia stores matrices by columns, so neighbouring threads take neighbouring
# rows. The time includes copying A and B to the GPU and C back.
using CUDA
include("common.jl")

function matmul_kernel_cuda!(C, A, B)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    if i <= size(C, 1) && j <= size(C, 2)
        @inbounds C[i, j] = matmul_entry(A, B, i, j)
    end
    return nothing
end

function matmul_gpu!(C, A, B)
    n = size(A, 1)
    dA, dB = CuArray(A), CuArray(B)
    dC = similar(dA)
    @cuda threads=(16, 16) blocks=(cld(n, 16), cld(n, 16)) matmul_kernel_cuda!(dC, dA, dB)
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
matrix_report("GPU (CUDA.jl kernel, $(CUDA.name(CUDA.device())))", A, B, C, seconds) || exit(1)
