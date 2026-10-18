# Matrix product C = A B on the GPU with Dagger.Kernel: Dagger launches the KernelAbstractions kernel
# as a task in the scope of the first CUDA GPU on arrays created with Dagger.@mutable (needed for
# in-place kernels), with KernelAbstractions' default workgroup size (Dagger.Kernel has no option for it;
# the product took about 1.6 times as long as with 16 x 16). The time includes copying A and B to the GPU
# and C back, which fetch returns as a new host array.
using CUDA, Dagger, KernelAbstractions
include("common.jl")

@kernel function matmul_kernel!(C, A, B)
    i, j = @index(Global, NTuple)
    @inbounds C[i, j] = matmul_entry(A, B, i, j)
end

function matmul_gpu(A, B)
    gpu = Dagger.scope(cuda_gpu=1)
    Dagger.with_options(; scope=gpu) do
        # On the GPU of the kernel's scope; without a scope, @mutable may use another GPU of the instance.
        dA, dB = Dagger.@mutable(scope=gpu, CuArray(A)), Dagger.@mutable(scope=gpu, CuArray(B))
        dC = Dagger.@mutable scope=gpu CuArray{Float32}(undef, size(A))
        fetch(Dagger.@spawn Dagger.Kernel(matmul_kernel!)(dC, dA, dB; ndrange=size(A)))
        Dagger.gpu_synchronize(:CUDA)
        return Array(fetch(dC))
    end
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
A, B = matrix_init(MATRIX_N)
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_gpu(A, B))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed C = matmul_gpu(A, B)
matrix_report("GPU (Dagger.Kernel, $(CUDA.name(CUDA.device())))", A, B, C, seconds) || exit(1)
