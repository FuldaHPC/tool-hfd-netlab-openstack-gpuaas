# pi and the matrix product C = A B, each sequential, with OpenMP on the CPU threads and offloaded to the GPU.
# gcc-offload is a wrapper for: gcc -fopenmp -foffload=nvptx-none -foffload-options=-fcf-protection=none -foffload-options=-Wa,--no-verify -fno-stack-protector

# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
CPU ?= -march=x86-64-v3
PROGRAMS = pi_seq pi_cpu pi_gpu matmul_seq matmul_cpu matmul_gpu

all: $(PROGRAMS)

%_seq: %_seq.c common.h
	gcc -O3 $(CPU) -o $@ $<

%_cpu: %_cpu.c common.h
	gcc -O3 $(CPU) -fopenmp -o $@ $<

%_gpu: %_gpu.c common.h
	gcc-offload -O3 $(CPU) -o $@ $<

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
