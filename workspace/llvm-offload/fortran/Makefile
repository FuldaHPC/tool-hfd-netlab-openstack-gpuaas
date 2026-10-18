# pi and the matrix product C = A B in Fortran with LLVM Flang: sequential, with OpenMP on the CPU
# threads, and offloaded to the GPU with OpenMP (Flang cannot offload OpenACC yet). The sources are the
# same as in ../../nvhpc. common.f90 holds the sizes and the checks.
# First load the modules: module load llvm cuda (as for Clang, see ../README.md).
# --offload-arch=native compiles for the GPU of this instance (sm_86 on an RTX 3070 or 3080).

OFFLOAD_ARCH ?= native
# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
CPU ?= -march=x86-64-v3
PROGRAMS = pi_seq pi_cpu pi_gpu matmul_seq matmul_cpu matmul_gpu

all: $(PROGRAMS)

common.o: common.f90
	flang -O3 $(CPU) -c $<

%_seq: %_seq.f90 common.o
	flang -O3 $(CPU) -o $@ $< common.o

%_cpu: %_cpu.f90 common.o
	flang -O3 $(CPU) -fopenmp -o $@ $< common.o

%_gpu: %_gpu.f90 common.o
	flang -O3 $(CPU) -fopenmp --offload-arch=$(OFFLOAD_ARCH) -o $@ $< common.o

run: all
	./pi_seq
	./pi_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./pi_gpu
	./matmul_seq
	./matmul_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./matmul_gpu

clean:
	rm -f $(PROGRAMS) *.o *.mod

.PHONY: all run clean
