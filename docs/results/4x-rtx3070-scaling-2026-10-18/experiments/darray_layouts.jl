# DArray matrix product with different block layouts on all GPUs (n from ARGS): 2 x 2 blocks as in the
# workspace program, one column block of B per GPU with A in one piece, and g x g blocks.
using CUDA, Dagger
include("common.jl")
const N = parse(Int, ARGS[1])
g = CUDA.ndevices()
layouts = [("2x2 blocks", Blocks(cld(N, 2), cld(N, 2)), Blocks(cld(N, 2), cld(N, 2))),
           ("A whole, B in $g column blocks", Blocks(N, N), Blocks(N, cld(N, g))),
           ("$(g)x$g blocks", Blocks(cld(N, g), cld(N, g)), Blocks(cld(N, g), cld(N, g)))]
A, B = matrix_init(N)
for (name, ba, bb) in layouts
    f() = Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:)) do
        collect(DArray(A, ba) * DArray(B, bb))
    end
    warm_up(f)
    GC.gc()
    times = Float64[]
    local C
    for _ in 1:3
        push!(times, @elapsed C = f())
    end
    matrix_report("$name, runs $(join(round.(times, digits=3), " ")), median", A, B, C, sort(times)[2])
end
