# pi and the matrix product C = A B: sequential, with MPI on the CPU cores, and with a CUDA kernel per rank on
# the GPU (CUDA-aware MPI moves the data directly between GPU buffers).
# First load the module: module load openmpi (it also loads cuda, so nvcc is in the PATH).

# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
CPU ?= -march=x86-64-v3
NP ?= $(shell nproc)
# The GPU programs run one rank per GPU (at least one): more ranks would make some GPUs do the work of two.
NP_GPU ?= $(shell n=$$(nvidia-smi -L 2>/dev/null | grep -c '^GPU'); echo $$(( n > 0 ? n : 1 )))
CUDA_ARCH ?= native
MPIRUN = mpirun --oversubscribe -np $(NP)
MPIRUN_GPU = mpirun --oversubscribe -np $(NP_GPU)
PROGRAMS = pi_seq pi_cpu pi_gpu matmul_seq matmul_cpu matmul_gpu

all: $(PROGRAMS)

# The sequential programs use no MPI; mpicc is GCC here.
%_seq: %_seq.c common.h
	mpicc -O3 $(CPU) -o $@ $<

%_cpu: %_cpu.c common.h
	mpicc -O3 $(CPU) -o $@ $<

# nvcc compiles the CUDA kernels and lets mpicxx compile and link the host code with MPI.
%_gpu: %_gpu.cu common.h
	nvcc -ccbin mpicxx -O3 -Xcompiler $(CPU) -arch=$(CUDA_ARCH) -o $@ $<

run: all
	./pi_seq
	$(MPIRUN) ./pi_cpu
	$(MPIRUN_GPU) ./pi_gpu
	./matmul_seq
	$(MPIRUN) ./matmul_cpu
	$(MPIRUN_GPU) ./matmul_gpu

clean:
	rm -f $(PROGRAMS)

.PHONY: all run clean
