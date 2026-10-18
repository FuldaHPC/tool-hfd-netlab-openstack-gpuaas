# pi and the matrix product C = A B in Fortran with gfortran: sequential, with OpenMP on the CPU
# threads, and offloaded to the GPU with OpenMP and with OpenACC. common.f90 holds the sizes and checks.
# gfortran-offload is a wrapper like gcc-offload (see ../README.md).

# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
CPU ?= -march=x86-64-v3
PROGRAMS = pi_seq pi_cpu pi_gpu pi_acc matmul_seq matmul_cpu matmul_gpu matmul_acc

all: $(PROGRAMS)

common.o: common.f90
	gfortran -O3 $(CPU) -c $<

%_seq: %_seq.f90 common.o
	gfortran -O3 $(CPU) -o $@ $< common.o

%_cpu: %_cpu.f90 common.o
	gfortran -O3 $(CPU) -fopenmp -o $@ $< common.o

%_gpu: %_gpu.f90 common.o
	gfortran-offload -O3 $(CPU) -o $@ $< common.o

%_acc: %_acc.f90 common.o
	gfortran-offload -O3 $(CPU) -fopenacc -o $@ $< common.o

run: all
	./pi_seq
	./pi_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./pi_gpu
	./pi_acc
	./matmul_seq
	./matmul_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./matmul_gpu
	./matmul_acc

clean:
	rm -f $(PROGRAMS) *.o *.mod

.PHONY: all run clean
