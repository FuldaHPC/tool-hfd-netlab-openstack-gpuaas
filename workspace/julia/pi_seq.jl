# pi with the midpoint rule in a plain loop on one CPU thread, as reference for pi_cpu.jl, which
# distributes the same loop over Dagger tasks.
include("common.jl")

function pi_seq(n)
    h = 1 / n
    s = 0.0
    for i in 0:n-1
        x = (i + 0.5) * h
        s += 4 / (1 + x * x)
    end
    return h * s
end

# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> pi_seq(PI_STEPS))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed approx = pi_seq(PI_STEPS)
pi_report("CPU sequential (Julia)", approx, PI_STEPS, seconds) || exit(1)
