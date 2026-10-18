# pi with the midpoint rule on the GPUs with Dagger's DArray: the chunk numbers become a DArray with one
# block per GPU in GPU memory, spread over all CUDA GPUs of the instance (scope cuda_gpus=:), map applies
# pi_chunk_sum on each block as GPU array code, and sum adds everything up. (A fixed four blocks left one GPU
# with twice the work on three GPUs.)
using CUDA, Dagger
include("common.jl")

function pi_gpu(n, chunks)
    h, size = 1 / n, cld(n, chunks)
    Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:)) do
        D = DArray(collect(0:chunks-1), Blocks(cld(chunks, CUDA.ndevices())))
        return h * sum(map(c -> pi_chunk_sum(c, size, n, h), D))
    end
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_gpu(PI_STEPS, PI_CHUNKS))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed approx = pi_gpu(PI_STEPS, PI_CHUNKS)
pi_report("GPU (Dagger DArray on $(CUDA.ndevices()) GPU$(CUDA.ndevices() > 1 ? "s" : ""), $(CUDA.name(CUDA.device())))", approx, PI_STEPS, seconds) || exit(1)
