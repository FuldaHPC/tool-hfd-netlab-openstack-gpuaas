# Shared by the Julia examples of this folder: problem sizes, test matrices and the result checks,
# so that the programs themselves only show the programming model.
using Printf

const PI_STEPS = 1_000_000_000  # steps of the midpoint rule for pi
const PI_FLOPS_PER_STEP = 6     # x = (i + 0.5) * h; s += 4 / (1 + x * x): 3 adds, 2 multiplies, 1 division
const MATRIX_N = 2048           # the matrices are MATRIX_N x MATRIX_N Float32
const PI_CHUNKS = 2^20          # GPU versions of pi: each GPU thread sums one chunk of the steps

# Warm-up with the full problem: runs f() until two runs in a row compile no code (at most 10 runs).
# Julia compiles code when a run first needs it, and with Dagger which code that is depends on timing
# (once, a third full-size run still compiled for 0.4 s), so a fixed number of warm-up runs is not enough.
function warm_up(f)
    quiet = 0
    for _ in 1:10
        quiet = (@timed f()).compile_time == 0 ? quiet + 1 : 0
        quiet == 2 && return
    end
end

# pi is the integral of 4/(1+x^2) over [0,1]; the midpoint rule sums 4/(1+x^2) at x = (i+0.5)/n.
# Rounding makes the error about 1e-13; one missing step would add at least 2e-9.
function pi_report(label, approx, n, seconds)
    error = abs(approx - pi)
    @printf("%s: pi = %.12f (error %.1e) in %.3f s (%.1f GFLOP/s)\n", label, approx, error, seconds,
            PI_FLOPS_PER_STEP * n / seconds / 1e9)
    flush(stdout)  # at once: once, with output to a file, a Dagger program ended without its line
    return error < 1e-9
end

# Small integers, so that every product and sum in C = A B is exact in single precision
# (indices from 0 as in the other languages).
matrix_init(n) = ([Float32((i + 2j) % 8) for i in 0:n-1, j in 0:n-1], [Float32((3i + j) % 8) for i in 0:n-1, j in 0:n-1])

# Checks C x == A (B x) for x = (1, 2, ..., n) in O(n^2); exact, as all values are integers below 2^53.
# x has no zero, so that every element of C counts.
function matrix_report(label, A, B, C, seconds)
    n = size(C, 1)
    x = collect(1.0:n)
    ok = Float64.(C) * x == Float64.(A) * (Float64.(B) * x)
    @printf("%s: %dx%d matrix product %s in %.3f s (%.1f GFLOP/s)\n", label, n, n, ok ? "correct" : "WRONG", seconds,
            2.0 * n^3 / seconds / 1e9)
    flush(stdout)
    return ok
end

# Sum of the midpoint values in chunk c (from 0) of the n steps; every GPU version of pi uses it, so they
# all do the same work per GPU thread.
@inline function pi_chunk_sum(c, size, n, h)
    s = 0.0
    for i in c*size:min((c + 1) * size, n)-1
        x = (i + 0.5) * h
        s += 4 / (1 + x * x)
    end
    return s
end

# Element (i, j) of C = A B; every hand-written GPU kernel for the matrix product uses it.
@inline function matmul_entry(A, B, i, j)
    s = zero(eltype(A))
    for k in axes(A, 2)
        @inbounds s += A[i, k] * B[k, j]
    end
    return s
end
