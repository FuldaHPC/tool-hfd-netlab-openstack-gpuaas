# Matrix product C = A B on the GPUs with Dagger's DArray: A and B become DArrays of 2 x 2 blocks in GPU
# memory, spread over all CUDA GPUs of the instance (scope cuda_gpus=:), and A * B multiplies them block
# by block with cuBLAS (a tuned library instead of the simple algorithm); collect copies C back into a
# new host array.
using CUDA, Dagger
include("common.jl")

function matmul_gpu(A, B)
    blocks = Blocks(cld(size(A, 1), 2), cld(size(A, 2), 2))
    Dagger.with_options(; scope=Dagger.scope(cuda_gpus=:)) do
        return collect(DArray(A, blocks) * DArray(B, blocks))
    end
end

CUDA.functional() || error("CUDA.jl cannot use the GPU")
A, B = matrix_init(MATRIX_N)
# Warm-up with the full problem until Julia compiles nothing more (see common.jl).
warm_up(() -> matmul_gpu(A, B))
# Collects the warm-up's garbage now, so that no garbage collection falls into the timed run.
GC.gc()
seconds = @elapsed C = matmul_gpu(A, B)
matrix_report("GPU (Dagger DArray on $(CUDA.ndevices()) GPU$(CUDA.ndevices() > 1 ? "s" : ""): A * B with cuBLAS, $(CUDA.name(CUDA.device())))", A, B, C, seconds) || exit(1)
