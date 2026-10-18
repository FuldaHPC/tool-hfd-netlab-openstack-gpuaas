# pi and the matrix product C = A B in Chapel: serial for loops, forall loops on the CPU cores, and
# forall loops that become GPU kernels inside "on here.gpus[0]". Common.chpl holds the sizes and the result checks.
# First load the module: module load chapel

PROGRAMS = pi_seq pi_cpu pi_gpu pi_gpus matmul_seq matmul_cpu matmul_gpu matmul_gpus

all: $(PROGRAMS)

%: %.chpl Common.chpl
	chpl --fast -o $@ $<

run: all
	./pi_seq
	./pi_cpu
	./pi_gpu
	./pi_gpus
	./matmul_seq
	./matmul_cpu
	./matmul_gpu
	./matmul_gpus

clean:
	rm -f $(PROGRAMS) $(PROGRAMS:=_real)

.PHONY: all run clean
