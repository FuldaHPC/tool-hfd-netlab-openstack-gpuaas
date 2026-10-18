# pi with the midpoint rule on the GPUs with Dagger's Datadeps: inside spawn_datadeps, one task per GPU
# of the instance (scope cuda_gpus=:) is marked Out(its part of the chunk sums); Dagger moves that part
# of the host array to the task's GPU, where the task launches the KernelAbstractions kernel (workgroups
# of 256), and copies it back at the end of the region; sum adds the chunk sums on the CPU.
using CUDA, Dagger, KernelAbstractions
include("common.jl")

@kernel function chunk_kernel!(parts, first, size, n, h)
    c = @index(Global, Linear)
    @inbounds parts[c] = pi_chunk_sum(first + c - 1, size, n, h)
end

function chunk_task!(parts, first, size, n, h)  # parts holds the sums of the chunks first, first+1, ...
    backend = Dagger.gpu_kernel_backend()
    chunk_kernel!(backend, 256)(parts, first, size, n, h; ndrange=length(parts))
    KernelAbstractions.synchronize(backend)
    return nothing
end

chunk_firsts(chunks, g) = [k * chunks ÷ g for k in 0:g]

# parts holds one array of chunk sums per GPU.
function pi_gpu!(parts, n, chunks)
    h, size, g = 1 / n, cld(n, chunks), length(parts)
    firsts = chunk_firsts(chunks, g)
    Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:)) do
        Dagger.spawn_datadeps() do
            for k in 1:g
                Dagger.@spawn chunk_task!(Out(parts[k]), firsts[k], size, n, h)
            end
        end
    end
    return h * sum(sum, parts)
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
g = CUDA.ndevices()
# One array of chunk sums per GPU (Dagger's Datadeps does not move views of arrays to a GPU and back),
# written now, so that its page faults do not fall into the timed run.
firsts = chunk_firsts(PI_CHUNKS, g)
parts = [zeros(firsts[k+1] - firsts[k]) for k in 1:g]
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_gpu!(parts, PI_STEPS, PI_CHUNKS))
# Clears the warm-up's sums, so that the check below sees the timed run's, and collects its garbage now,
# so that no garbage collection falls into the timed run.
foreach(p -> fill!(p, 0), parts)
GC.gc()
seconds = @elapsed approx = pi_gpu!(parts, PI_STEPS, PI_CHUNKS)
pi_report("GPU (Dagger Datadeps on $g GPU$(g > 1 ? "s" : ""), $(CUDA.name(CUDA.device())))", approx, PI_STEPS, seconds) || exit(1)
