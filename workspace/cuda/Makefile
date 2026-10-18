# pi and the matrix product C = A B, each sequential in plain C++ on one CPU core and in a CUDA kernel
# on the GPU (CUDA has no parallel CPU version).
# nvcc is in the PATH through /etc/profile.d/cuda.sh (log in again after the installation);
# CUDA_ARCH=native compiles for the GPU of this instance.

# CPU: x86-64-v3 (AVX2, FMA), like the Spack builds; on the cloud hosts as fast as -march=native, and the
# programs still run when a snapshot is started on another host.
CPU ?= -march=x86-64-v3
CUDA_ARCH ?= native
PROGRAMS = pi_seq pi_gpu matmul_seq matmul_gpu

all: $(PROGRAMS)

%_seq: %_seq.cpp common.h
	g++ -O3 $(CPU) -o $@ $<

%_gpu: %_gpu.cu common.h
	nvcc -O3 -Xcompiler $(CPU) -arch=$(CUDA_ARCH) -o $@ $<

run: all
	./pi_seq
	./pi_gpu
	./matmul_seq
	./matmul_gpu

clean:
	rm -f $(PROGRAMS)

.PHONY: all run clean
