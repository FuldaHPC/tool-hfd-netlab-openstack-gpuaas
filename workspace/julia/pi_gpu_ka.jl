# pi with the midpoint rule on the GPU with a KernelAbstractions.jl kernel, without Dagger: the same
# kernel would run on CPU threads or other GPUs with another backend; here CUDABackend(). Each GPU
# thread sums one chunk of the steps (workgroups of 256, as the @cuda kernel), and sum adds the chunk
# sums on the GPU.
using CUDA, KernelAbstractions
include("common.jl")

@kernel function chunk_kernel!(parts, size, n, h)
    c = @index(Global, Linear)
    @inbounds parts[c] = pi_chunk_sum(c - 1, size, n, h)
end

function pi_gpu(n, chunks)
    backend = CUDABackend()
    h, size = 1 / n, cld(n, chunks)
    parts = KernelAbstractions.allocate(backend, Float64, chunks)
    chunk_kernel!(backend, 256)(parts, size, n, h; ndrange=chunks)
    return h * sum(parts)
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_gpu(PI_STEPS, PI_CHUNKS))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed approx = pi_gpu(PI_STEPS, PI_CHUNKS)
pi_report("GPU (KernelAbstractions, $(CUDA.name(CUDA.device())))", approx, PI_STEPS, seconds) || exit(1)
