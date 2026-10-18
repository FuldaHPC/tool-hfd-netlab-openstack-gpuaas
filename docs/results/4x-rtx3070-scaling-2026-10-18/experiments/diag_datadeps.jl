# Diagnosis of matmul_gpu_dagger_datadeps.jl: which GPU each task runs on and when, and how the time splits
# into preparing the per-GPU arrays, the Datadeps region and copying the result back.
# Usage: julia [-t N] diag_datadeps.jl N auto|pinned
#   auto:   as in the workspace program (scope cuda_gpus=:, Dagger chooses the GPU of each task)
#   pinned: task k gets the scope cuda_gpu=k
using CUDA, Dagger, KernelAbstractions
include("common.jl")
const N = parse(Int, ARGS[1])
const PIN = get(ARGS, 2, "auto") == "pinned"

@kernel function matmul_kernel!(C, A, B)
    i, j = @index(Global, NTuple)
    @inbounds C[i, j] = matmul_entry(A, B, i, j)
end

const EVENTS = Tuple{Int,Float64,Float64}[]
const EVENTS_LOCK = ReentrantLock()
const T0 = Ref(0.0)

function matmul_task!(C, A, B)
    start = time() - T0[]
    backend = Dagger.gpu_kernel_backend()
    matmul_kernel!(backend, (16, 16))(C, A, B; ndrange=size(C))
    KernelAbstractions.synchronize(backend)
    lock(() -> push!(EVENTS, (CUDA.deviceid(CUDA.device()), start, time() - T0[])), EVENTS_LOCK)
    return nothing
end

function matmul_gpu!(C, A, B)
    n, g = size(C, 2), CUDA.ndevices()
    blocks = [k*n÷g+1:(k+1)*n÷g for k in 0:g-1]
    T0[] = time()
    prepare = @elapsed begin
        Bs = g == 1 ? [B] : [B[:, cols] for cols in blocks]
        Cs = g == 1 ? [C] : [similar(C, n, length(cols)) for cols in blocks]
    end
    region = @elapsed Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:)) do
        Dagger.spawn_datadeps() do
            for k in 1:g
                if PIN
                    Dagger.@spawn scope=Dagger.scope(cuda_gpu=k) matmul_task!(Out(Cs[k]), In(A), In(Bs[k]))
                else
                    Dagger.@spawn matmul_task!(Out(Cs[k]), In(A), In(Bs[k]))
                end
            end
        end
    end
    gather = @elapsed (g == 1 || foreach((cols, Ck) -> copyto!(view(C, :, cols), Ck), blocks, Cs))
    return prepare, region, gather
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
println("threads $(Threads.nthreads()), GPUs $(CUDA.ndevices()), n $N, ", PIN ? "pinned" : "auto")
A, B = matrix_init(N)
C = zeros(Float32, N, N)
for _ in 1:3
    matmul_gpu!(similar(A), A, B)  # warm-up
end
GC.gc()
for run in 1:3
    empty!(EVENTS)
    total = @elapsed (parts = matmul_gpu!(C, A, B))
    tasks = join(["GPU $d $(round(s, digits=3))-$(round(e, digits=3)) s" for (d, s, e) in sort(EVENTS, by=x -> x[2])], ", ")
    println("run $run: total $(round(total, digits=3)) s = prepare $(round(parts[1], digits=3)) + region ",
            "$(round(parts[2], digits=3)) + gather $(round(parts[3], digits=3)); kernels: $tasks")
end
matrix_report("check", A, B, C, 1.0)
