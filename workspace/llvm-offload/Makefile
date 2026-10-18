# pi and the matrix product C = A B, each sequential, with OpenMP on the CPU threads and offloaded to the GPU,
# compiled with Clang. The sources are the same as in gcc-offload/.
# First load the modules: module load llvm cuda (clang needs CUDA's ptxas for the GPU code). The
# programs get an rpath to LLVM's OpenMP runtime, so they also run without the modules.
# --offload-arch=native compiles for the GPU of this instance (sm_86 on an RTX 3070 or 3080).

# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
CPU ?= -march=x86-64-v3
OFFLOAD_ARCH ?= native
PROGRAMS = pi_seq pi_cpu pi_gpu matmul_seq matmul_cpu matmul_gpu

all: $(PROGRAMS)

%_seq: %_seq.c common.h
	clang -O3 $(CPU) -o $@ $<

%_cpu: %_cpu.c common.h
	clang -O3 $(CPU) -fopenmp -o $@ $<

%_gpu: %_gpu.c common.h
	clang -O3 $(CPU) -fopenmp --offload-arch=$(OFFLOAD_ARCH) -o $@ $<

run: all
	./pi_seq
	./pi_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./pi_gpu
	./matmul_seq
	./matmul_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./matmul_gpu

clean:
	rm -f $(PROGRAMS)

.PHONY: all run clean
