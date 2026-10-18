# Matrix product C = A B in plain loops on one CPU thread, as reference for matmul_cpu.jl, which
# distributes the columns over Dagger tasks. Julia stores matrices by columns, so the loop order j-k-i
# lets the innermost loop run along columns.
include("common.jl")

function matmul_seq!(C, A, B)
    n = size(C, 1)
    C .= 0
    @inbounds for j in 1:n, k in 1:n
        bkj = B[k, j]
        for i in 1:n
            C[i, j] += A[i, k] * bkj
        end
    end
    return C
end

A, B = matrix_init(MATRIX_N)
C = zeros(Float32, size(A))  # written once now, so that its page faults do not fall into the timed run
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_seq!(similar(A), A, B))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed matmul_seq!(C, A, B)
matrix_report("CPU sequential (Julia)", A, B, C, seconds) || exit(1)
