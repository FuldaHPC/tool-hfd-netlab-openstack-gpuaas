# pi with the midpoint rule on the GPU with CUDA.jl array programming, without Dagger: broadcasting
# pi_chunk_sum over the chunk numbers into a CuArray becomes a GPU kernel (one thread per chunk), and
# sum adds the chunk sums on the GPU.
using CUDA
include("common.jl")

function pi_gpu(n, chunks)
    h, size = 1 / n, cld(n, chunks)
    parts = CuArray{Float64}(undef, chunks)
    parts .= pi_chunk_sum.(0:chunks-1, size, n, h)
    return h * sum(parts)
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_gpu(PI_STEPS, PI_CHUNKS))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed approx = pi_gpu(PI_STEPS, PI_CHUNKS)
pi_report("GPU (CUDA.jl arrays, $(CUDA.name(CUDA.device())))", approx, PI_STEPS, seconds) || exit(1)
