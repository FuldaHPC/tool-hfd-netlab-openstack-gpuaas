# pi with the midpoint rule in Dagger tasks on the CPU threads: the steps are split into four tasks per
# thread. Start Julia with threads: julia -t auto pi_cpu.jl
using Dagger
include("common.jl")

function part(first, last, h)
    s = 0.0
    for i in first:last
        x = (i + 0.5) * h
        s += 4 / (1 + x * x)
    end
    return s
end

function pi_dagger(n, tasks)
    h = 1 / n
    parts = [Dagger.@spawn part(t * n ÷ tasks, (t + 1) * n ÷ tasks - 1, h) for t in 0:tasks-1]
    return h * sum(fetch.(parts))
end

tasks = 4 * Threads.nthreads()
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_dagger(PI_STEPS, tasks))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed approx = pi_dagger(PI_STEPS, tasks)
pi_report("CPU (Dagger, $(Threads.nthreads()) threads, $tasks tasks)", approx, PI_STEPS, seconds) || exit(1)
