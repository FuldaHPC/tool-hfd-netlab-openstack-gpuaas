# pi and the matrix product C = A B in Fortran with nvfortran (NVIDIA HPC SDK): sequential, with OpenMP
# on the CPU threads, and on the GPU with OpenMP, OpenACC, do concurrent (standard parallelism) and
# CUDA Fortran. common.f90 holds the sizes and the checks. First load the module: module load nvhpc
# GPU=ccnative compiles for the GPU of this instance (cc86 on an RTX 3070 or 3080).

GPU ?= ccnative
# CPU: x86-64-v3 (AVX2, FMA), like the other folders and the Spack builds; on the cloud hosts as fast as
# nvfortran's default -tp=native, and the programs still run when a snapshot is started on another host.
CPU ?= -tp=x86-64-v3
PROGRAMS = pi_seq pi_cpu pi_gpu pi_acc pi_stdpar pi_cuf matmul_seq matmul_cpu matmul_gpu matmul_acc matmul_stdpar matmul_cuf

all: $(PROGRAMS)

common.o: common.f90
	nvfortran -O3 $(CPU) -c $<

%_seq: %_seq.f90 common.o
	nvfortran -O3 $(CPU) -o $@ $< common.o

%_cpu: %_cpu.f90 common.o
	nvfortran -O3 $(CPU) -mp -o $@ $< common.o

%_gpu: %_gpu.f90 common.o
	nvfortran -O3 $(CPU) -mp=gpu -gpu=$(GPU) -o $@ $< common.o

%_acc: %_acc.f90 common.o
	nvfortran -O3 $(CPU) -acc=gpu -gpu=$(GPU) -o $@ $< common.o

%_stdpar: %_stdpar.f90 common.o
	nvfortran -O3 $(CPU) -stdpar=gpu -gpu=$(GPU) -o $@ $< common.o

%_cuf: %_cuf.cuf common.o
	nvfortran -O3 $(CPU) -cuda -gpu=$(GPU) -o $@ $< common.o

run: all
	./pi_seq
	./pi_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./pi_gpu
	./pi_acc
	./pi_stdpar
	./pi_cuf
	./matmul_seq
	./matmul_cpu
	OMP_TARGET_OFFLOAD=MANDATORY ./matmul_gpu
	./matmul_acc
	./matmul_stdpar
	./matmul_cuf

clean:
	rm -f $(PROGRAMS) *.o *.mod

.PHONY: all run clean
