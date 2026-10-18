# Matrix product C = A B on the GPUs with Dagger's Datadeps: inside spawn_datadeps, one task per GPU of
# the instance (scope cuda_gpus=:) gets Out(its columns of C), In(A) and In(the same columns of B) and
# launches the KernelAbstractions kernel (workgroups of 16 x 16); Dagger moves the inputs to the task's
# GPU and the result back at the end of the region (included in the time).
using CUDA, Dagger, KernelAbstractions
include("common.jl")

@kernel function matmul_kernel!(C, A, B)
    i, j = @index(Global, NTuple)
    @inbounds C[i, j] = matmul_entry(A, B, i, j)
end

function matmul_task!(C, A, B)
    backend = Dagger.gpu_kernel_backend()
    matmul_kernel!(backend, (16, 16))(C, A, B; ndrange=size(C))
    KernelAbstractions.synchronize(backend)
    return nothing
end

column_blocks(n, g) = [k*n÷g+1:(k+1)*n÷g for k in 0:g-1]

# Bs and Cs hold each task's columns of B and C (with one GPU, B and C themselves).
function matmul_gpu!(C, A, B, Bs, Cs)
    g = length(Bs)
    blocks = column_blocks(size(C, 2), g)
    g == 1 || foreach((cols, Bk) -> copyto!(Bk, view(B, :, cols)), blocks, Bs)
    Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:)) do
        Dagger.spawn_datadeps() do
            for k in 1:g
                Dagger.@spawn matmul_task!(Out(Cs[k]), In(A), In(Bs[k]))
            end
        end
    end
    g == 1 || foreach((cols, Ck) -> copyto!(view(C, :, cols), Ck), blocks, Cs)
    return C
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
g = CUDA.ndevices()
A, B = matrix_init(MATRIX_N)
C = zeros(Float32, size(A))  # written once now, so that its page faults do not fall into the timed run
# With several GPUs, every task gets its own columns of B and C as separate arrays (Dagger's Datadeps does
# not move views of arrays to a GPU and back), written now like C: new arrays inside the timed run made
# two GPUs slower than one (0.5 s of page faults at n = 8192).
Bs, Cs = g == 1 ? ([B], [C]) : ([zeros(Float32, MATRIX_N, length(cols)) for cols in column_blocks(MATRIX_N, g)],
                                [zeros(Float32, MATRIX_N, length(cols)) for cols in column_blocks(MATRIX_N, g)])
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_gpu!(C, A, B, Bs, Cs))
# Clears the warm-up's result, so that the check below sees the timed run's, and collects its garbage now,
# so that no garbage collection falls into the timed run.
foreach(X -> fill!(X, 0), (C, Cs...))
GC.gc()
seconds = @elapsed matmul_gpu!(C, A, B, Bs, Cs)
matrix_report("GPU (Dagger Datadeps on $g GPU$(g > 1 ? "s" : ""), $(CUDA.name(CUDA.device())))", A, B, C, seconds) || exit(1)
