# Matrix product C = A B in Dagger tasks on the CPU threads, each task computing a block of columns of C.
# Julia stores matrices by columns, so this is the counterpart of distributing rows in the C versions:
# with the loop order j-k-i the innermost loop runs along whole columns.
# Start Julia with threads: julia -t auto matmul_cpu.jl
using Dagger
include("common.jl")

function columns!(C, A, B, cols)
    n = size(C, 1)
    C[:, cols] .= 0
    @inbounds for j in cols, k in 1:n
        bkj = B[k, j]
        for i in 1:n
            C[i, j] += A[i, k] * bkj
        end
    end
    return nothing
end

function matmul_dagger!(C, A, B, tasks)
    n = size(C, 2)
    blocks = [t*n÷tasks+1:(t+1)*n÷tasks for t in 0:tasks-1]
    foreach(fetch, [Dagger.@spawn columns!(C, A, B, cols) for cols in blocks])
    return C
end

tasks = 4 * Threads.nthreads()
A, B = matrix_init(MATRIX_N)
C = zeros(Float32, size(A))  # written once now, so that its page faults do not fall into the timed run
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_dagger!(similar(A), A, B, tasks))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed matmul_dagger!(C, A, B, tasks)
matrix_report("CPU (Dagger, $(Threads.nthreads()) threads, $tasks tasks)", A, B, C, seconds) || exit(1)
