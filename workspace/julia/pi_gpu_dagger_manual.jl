# pi with the midpoint rule on the GPU with Dagger's manual approach: a Dagger task in the scope of the
# first CUDA GPU asks Dagger.gpu_kernel_backend() for the backend of its processor and launches the
# KernelAbstractions kernel itself. The program fails unless the task ran on the GPU.
using CUDA, Dagger, KernelAbstractions
include("common.jl")

@kernel function chunk_kernel!(parts, size, n, h)
    c = @index(Global, Linear)
    @inbounds parts[c] = pi_chunk_sum(c - 1, size, n, h)
end

function gpu_task(n, chunks)
    backend = Dagger.gpu_kernel_backend()
    h, size = 1 / n, cld(n, chunks)
    parts = KernelAbstractions.allocate(backend, Float64, chunks)
    chunk_kernel!(backend, 256)(parts, size, n, h; ndrange=chunks)
    return string(nameof(typeof(backend))), h * sum(parts)
end

pi_gpu(n, chunks) = Dagger.with_options(; scope=Dagger.scope(cuda_gpu=1)) do
    fetch(Dagger.@spawn gpu_task(n, chunks))
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_gpu(PI_STEPS, PI_CHUNKS))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed (backend, approx) = pi_gpu(PI_STEPS, PI_CHUNKS)
ok = pi_report("GPU (Dagger manual on $backend, $(CUDA.name(CUDA.device())))", approx, PI_STEPS, seconds)
(ok && occursin("CUDA", backend)) || exit(1)
