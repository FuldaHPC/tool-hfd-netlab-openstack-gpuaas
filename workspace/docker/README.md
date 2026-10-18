# Docker with the NVIDIA container toolkit

`run.sh` starts two Ubuntu containers:

- CPU: a container that reports the CPUs it sees.
- GPU: a container started with `--gpus all`, in which `nvidia-smi -L` lists the GPU.

```sh
./run.sh
```

User `ubuntu` is in the group `docker` and needs no sudo; any image from Docker Hub or NGC (e.g. `nvcr.io/nvidia/pytorch`) works the same way with `--gpus all`.
