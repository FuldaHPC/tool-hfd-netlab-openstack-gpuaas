# HFD NetLab GPUaaS

`gpuaas.py` starts GPU instances (NVIDIA RTX 3070/3080 via PCI passthrough) in the [NetLab OpenStack cloud](https://private-cloud.informatik.hs-fulda.de/) of Hochschule Fulda and installs the software you choose: CUDA, GPU offloading compilers, Docker and an HPC stack (Open MPI, Kokkos, HPX, Chapel, Julia, LLVM), with examples that compare the programming models.
You need an account in that cloud, macOS or Linux with `ssh`, and the HS Fulda network or the VPN profile "05)_AI-NetLab-Pro".

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv runs gpuaas.py with its dependencies
git clone https://github.com/FuldaHPC/tool-hfd-netlab-openstack-gpuaas.git
cd tool-hfd-netlab-openstack-gpuaas
./gpuaas.py start                                    # the first run asks for username and password
```

## Usage

`./gpuaas.py help`, or just `./gpuaas.py`, prints:

```text
usage: gpuaas.py [-h] [--debug] COMMAND ...

Start, list, log in to, follow, snapshot, shelve, rename and delete GPU instances (NVIDIA RTX
cards via PCI passthrough) on the OpenStack cloud of NetLab, Hochschule Fulda.

Endpoints and username come from clouds.yaml (cloud "openstack") without a password;
the password is asked for (or taken from OS_PASSWORD) when there is no valid login token
from an earlier run, i.e. about once an hour. See docs/guide.md.

positional arguments:
  COMMAND
    start     start a GPU instance (asks for name, flavor, software, image and key)
    list      list your instances and the project quota
    ssh       log in to an instance or run a command there
    logs      follow the installation log of an instance
    snapshot  save an instance as an image for the image menu
    shelve    shelve an instance: keep its disk, release its GPU
    unshelve  resume a shelved instance
    rename    rename an instance and its hostname
    delete    delete an instance and release its floating IP
    setup     create clouds.yaml (username and project, no password)
    logout    delete the saved login token
    help      show this help, or the help of a command

options:
  -h, --help  show this help message and exit
  --debug     log all OpenStack API calls

examples:
  gpuaas.py start                      asks for name, flavor, software, image and SSH key
  gpuaas.py start my-gpu --flavor g1-3080.medium --packages cuda,gcc-offload --yes
  gpuaas.py list                       your instances, days until shelving, project quota
  gpuaas.py ssh my-gpu                 log in (or run a command: gpuaas.py ssh my-gpu nvidia-smi)
  gpuaas.py logs my-gpu                follow the installation
  gpuaas.py snapshot my-gpu my-image   save a ready instance as an image for later starts
  gpuaas.py unshelve my-gpu            resume a shelved instance (ssh offers this too)
  gpuaas.py rename my-gpu new-name     rename the instance and its hostname
  gpuaas.py delete my-gpu              delete the instance and release its floating IP

setup:
  The first run asks for your OpenStack username and password, lets you choose your
  project and writes clouds.yaml next to gpuaas.py, without the password. Each login
  saves its token for one hour, so the password (or OS_PASSWORD) is asked for about
  once an hour; "gpuaas.py logout" deletes the token. "gpuaas.py setup" starts over.
  Off campus, connect to the VPN profile "05)_AI-NetLab-Pro" first.

"gpuaas.py help COMMAND" shows the options of a command; docs/guide.md has the details.
```

`./gpuaas.py help start` lists the software packages.
[docs/guide.md](docs/guide.md) has the details, [workspace/README.md](workspace/README.md) the examples and their results.

## License

MIT, see [LICENSE](LICENSE).

Based on [srieger1/hfd-netlab-openstack-gpuaas](https://github.com/srieger1/hfd-netlab-openstack-gpuaas/tree/d1903d98) by Sebastian Rieger.

## AI disclaimer

Most of the code, the examples and the documentation in this repository were written with the AI coding assistant Claude Code (Anthropic), directed by the maintainer and tested on the NetLab cloud.
Review them like any other code before you rely on them, and report problems as issues.
