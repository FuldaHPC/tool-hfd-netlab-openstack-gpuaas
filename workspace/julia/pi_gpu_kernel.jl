# pi with the midpoint rule on the GPU with a hand-written CUDA.jl kernel (@cuda), without Dagger: each
# GPU thread sums one chunk of the steps into a CuArray, and sum adds the chunk sums on the GPU.
using CUDA
include("common.jl")

function chunk_kernel_cuda!(parts, size, n, h)
    c = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if c <= length(parts)
        @inbounds parts[c] = pi_chunk_sum(c - 1, size, n, h)
    end
    return nothing
end

function pi_gpu(n, chunks)
    h, size = 1 / n, cld(n, chunks)
    parts = CuArray{Float64}(undef, chunks)
    @cuda threads=256 blocks=cld(chunks, 256) chunk_kernel_cuda!(parts, size, n, h)
    return h * sum(parts)
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_gpu(PI_STEPS, PI_CHUNKS))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed approx = pi_gpu(PI_STEPS, PI_CHUNKS)
pi_report("GPU (CUDA.jl kernel, $(CUDA.name(CUDA.device())))", approx, PI_STEPS, seconds) || exit(1)
