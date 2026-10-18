# Matrix product C = A B on the GPU with Dagger's manual approach: a Dagger task in the scope of the first
# CUDA GPU gets the backend of its processor from Dagger.gpu_kernel_backend(), allocates C there and
# launches the KernelAbstractions kernel. Dagger moves the task's array arguments A and B to that GPU
# before the task starts, and fetch moves C back as a new host array (all included in the time). The
# program fails unless the task ran on the GPU.
using CUDA, Dagger, KernelAbstractions
include("common.jl")

@kernel function matmul_kernel!(C, A, B)
    i, j = @index(Global, NTuple)
    @inbounds C[i, j] = matmul_entry(A, B, i, j)
end

function gpu_task(A, B)  # A and B are already on the GPU of this task's processor
    backend = Dagger.gpu_kernel_backend()
    C = KernelAbstractions.allocate(backend, Float32, size(A))
    matmul_kernel!(backend, (16, 16))(C, A, B; ndrange=size(C))
    return string(nameof(typeof(backend))), C
end

matmul_gpu(A, B) = Dagger.with_options(; scope=Dagger.scope(cuda_gpu=1)) do
    fetch(Dagger.@spawn gpu_task(A, B))
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
A, B = matrix_init(MATRIX_N)
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_gpu(A, B))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed (backend, C) = matmul_gpu(A, B)
ok = matrix_report("GPU (Dagger manual on $backend, $(CUDA.name(CUDA.device())))", A, B, C, seconds)
(ok && occursin("CUDA", backend)) || exit(1)
