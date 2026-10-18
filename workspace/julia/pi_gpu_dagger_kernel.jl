# pi with the midpoint rule on the GPU with Dagger.Kernel: Dagger launches the KernelAbstractions
# kernel as a task in the scope of the first CUDA GPU and picks the backend from the processor the task
# runs on, with KernelAbstractions' default workgroup size (Dagger.Kernel has no option for it). The
# chunk sums go into an array that Dagger.@mutable creates for in-place use; sum adds them.
using CUDA, Dagger, KernelAbstractions
include("common.jl")

@kernel function chunk_kernel!(parts, size, n, h)
    c = @index(Global, Linear)
    @inbounds parts[c] = pi_chunk_sum(c - 1, size, n, h)
end

function pi_gpu(n, chunks)
    h, size = 1 / n, cld(n, chunks)
    gpu = Dagger.scope(cuda_gpu=1)
    Dagger.with_options(; scope=gpu) do
        # On the GPU of the kernel's scope; without a scope, @mutable may use another GPU of the instance.
        parts = Dagger.@mutable scope=gpu CuArray{Float64}(undef, chunks)
        fetch(Dagger.@spawn Dagger.Kernel(chunk_kernel!)(parts, size, n, h; ndrange=chunks))
        Dagger.gpu_synchronize(:CUDA)
        return h * sum(fetch(parts))
    end
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_gpu(PI_STEPS, PI_CHUNKS))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed approx = pi_gpu(PI_STEPS, PI_CHUNKS)
pi_report("GPU (Dagger.Kernel, $(CUDA.name(CUDA.device())))", approx, PI_STEPS, seconds) || exit(1)
