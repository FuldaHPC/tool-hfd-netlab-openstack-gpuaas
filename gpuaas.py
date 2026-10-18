#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["openstacksdk>=4", "pyyaml>=6", "questionary>=2"]
# ///
"""Start, list, log in to, follow, snapshot, shelve, rename and delete GPU instances (NVIDIA RTX
cards via PCI passthrough) on the OpenStack cloud of NetLab, Hochschule Fulda.

Endpoints and username come from clouds.yaml (cloud "openstack") without a password;
the password is asked for (or taken from OS_PASSWORD) when there is no valid login token
from an earlier run, i.e. about once an hour. See docs/guide.md.
"""

import argparse
import base64
import contextlib
import copy
import datetime
import getpass
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import openstack
import openstack.config
import questionary
import urllib3
import yaml
from keystoneauth1 import exceptions as auth_exceptions
from keystoneauth1 import session as keystone_session
from keystoneauth1.identity import v3 as keystone_v3
from openstack import exceptions

# clouds.yaml has verify: false, because the cloud's service endpoints use certificates of its internal CA
# (see clouds.yaml.example); urllib3 would otherwise warn about it on every API request.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HERE = Path(__file__).resolve().parent
CLOUDS_YAML = HERE / "clouds.yaml"
CLOUDS_TEMPLATE = HERE / "clouds.yaml.example"
PACKAGE_DIR = HERE / "cloud-init" / "packages"
WORKSPACE_DIR = HERE / "workspace"
NEW_KEY_FILE = Path.home() / ".ssh" / "gpuaas_ed25519"

CLOUD = "openstack"
DEFAULT_IMAGE = "ubuntu-26.04-resolute-server-cloud-image-amd64v3"
DEFAULT_FLAVOR = "g1-3070.xlarge"
SUPPORTED_GPUS = ("3070", "3080")  # GPU models (pci_passthrough alias); the RTX 2080 Ti is too old
DEFAULT_BURN_SECONDS = 600
SSH_USER = "ubuntu"
HOME_DIR = f"/home/{SSH_USER}"
REMOTE_WORKSPACE = f"{HOME_DIR}/workspace"
PHASE_FILE = "/var/lib/gpuaas/phase"  # installing, rebooting, building <package>, finishing, ready, failed: <package>
WORKSPACE_TAR = "/var/lib/gpuaas/workspace.tar.xz"
CHECK_WORKSPACE = "/var/lib/gpuaas/workspace"  # unchanged copy, the checks run copies of it
CHECK_DIR = "/usr/local/lib/gpuaas/checks"  # one script per check, run in parallel by gpuaas-finish
USERDATA_LIMIT = 65535  # Nova's limit for base64-encoded user data
SSH_WAIT_SECONDS = 300
READY_POLL_SECONDS = 15
SHELVE_AFTER_DAYS = 7
OTHER = "<other>"  # menu value for "enter another image name"
TOKEN_CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "gpuaas" / "token.json"
# Longest API wait per command in minutes (start: the wizard and the wait for ACTIVE). A saved login token is
# only reused if it stays valid a few minutes longer, so that it cannot expire during the wait.
WAIT_MINUTES = {"start": 20, "snapshot": 40, "shelve": 40, "unshelve": 20}
TOKEN_MINUTES = {command: minutes + 5 for command, minutes in WAIT_MINUTES.items()}
DEFAULT_TOKEN_MINUTES = 5
SHELVED = ("SHELVED", "SHELVED_OFFLOADED")
BASE_DISK_GB = 4  # Ubuntu with its updates, before the packages (their disk: in the package files)

# The check of a package without its own: its run.sh as user ubuntu in a fresh copy of the unchanged
# workspace (not runuser -l: its PAM session would move the output out of the gpuaas-finish journal).
CHECK_WORKSPACE_SCRIPT = f"""\
#!/bin/sh
# gpuaas-check-workspace PACKAGE: runs workspace/PACKAGE/run.sh in a fresh copy (installed by gpuaas.py).
exec runuser -u {SSH_USER} -- env HOME={HOME_DIR} bash -lc \\
  'd=$(mktemp -d) && cp -r {CHECK_WORKSPACE}/"$1" "$d" && cd "$d/$1" && ./run.sh; s=$?; rm -rf "$d"; exit $s' bash "$1"
"""

FINISH_UNIT = """\
[Unit]
Description=gpuaas: check the installed software, mark the instance ready, run jobs
Wants=network-online.target
After=network-online.target docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/gpuaas-finish
TimeoutStartSec=infinity

[Install]
WantedBy=multi-user.target
"""


# ---------------------------------------------------------------- connection and helpers

_password = None  # entered during setup, so it is not asked twice in the same run


def interactive(args=None):
    """True if questions can be asked: in a terminal, and without --yes."""
    return not getattr(args, "yes", False) and sys.stdin.isatty() and sys.stdout.isatty()


def ask_password(_prompt=None):
    """Called by openstacksdk when clouds.yaml has no password (getpass reads it from the terminal)."""
    password = _password or os.environ.get("OS_PASSWORD")
    if password:
        return password
    if not sys.stdin.isatty():
        sys.exit("No password: run gpuaas.py in a terminal or set OS_PASSWORD.")
    return getpass.getpass("OpenStack password: ")


def setup_clouds_yaml():
    """Create clouds.yaml from the template: ask for username and password, choose the project."""
    global _password
    if not interactive():
        sys.exit(f"No clouds.yaml: run gpuaas.py once in a terminal to create it, or copy {CLOUDS_TEMPLATE.name} "
                 "to clouds.yaml and fill in username and project_id.")
    if CLOUDS_YAML.exists() and not ask(questionary.confirm(f"Replace {tilde(CLOUDS_YAML)}?", default=False)):
        sys.exit("Aborted.")
    config = yaml.safe_load(CLOUDS_TEMPLATE.read_text())
    cloud = config["clouds"][CLOUD]
    auth = cloud["auth"]
    print(f"Creating {tilde(CLOUDS_YAML)} from {CLOUDS_TEMPLATE.name}; your password is not stored.")
    auth["username"] = ask(questionary.text("OpenStack username:", default=os.environ.get("OS_USERNAME", ""),
                                            validate=lambda v: bool(v.strip()) or "Enter your username.")).strip()
    _password = ask_password()
    # Log in without a project to list the projects this user may use.
    login = keystone_v3.Password(auth_url=auth["auth_url"], username=auth["username"], password=_password,
                                 user_domain_name=auth["user_domain_name"], unscoped=True)
    session = keystone_session.Session(auth=login, verify=cloud.get("verify", True))
    projects = session.get(auth["auth_url"].rstrip("/") + "/v3/auth/projects").json()["projects"]
    if not projects:
        sys.exit(f"User {auth['username']} has no OpenStack project; ask the NetLab admins.")
    by_id = {p["id"]: p for p in sorted(projects, key=lambda p: p["name"])}
    project_id = next(iter(by_id)) if len(by_id) == 1 else ask(questionary.select(
        "Project:", choices=[questionary.Choice(p["name"], value=i) for i, p in by_id.items()]))
    auth["project_id"], auth["project_name"] = project_id, by_id[project_id]["name"]
    CLOUDS_YAML.write_text(f"# Created by gpuaas.py from {CLOUDS_TEMPLATE.name}; no password (asked about once an hour).\n"
                           + yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {tilde(CLOUDS_YAML)} for user {auth['username']}, project {auth['project_name']}.")


def connect(minutes=DEFAULT_TOKEN_MINUTES):
    """Connect with the saved login token if it stays valid for `minutes`, otherwise with the password."""
    # First run: no clouds.yaml next to the script and no cloud "openstack" elsewhere.
    if not CLOUDS_YAML.exists() and CLOUD not in openstack.config.OpenStackConfig().get_cloud_names():
        setup_clouds_yaml()
    # Find clouds.yaml next to this script even when called from another directory.
    if CLOUDS_YAML.exists():
        os.environ.setdefault("OS_CLIENT_CONFIG_FILE", str(CLOUDS_YAML))
    config = openstack.config.OpenStackConfig(pw_func=ask_password)
    settings = config.get_one(cloud=CLOUD, validate=False).config  # validate=False: no password prompt yet
    auth = settings["auth"]
    token = cached_token(auth, minutes)
    if token:
        # Built by hand: the auth section of clouds.yaml holds password options that a token login rejects.
        login = keystone_v3.Token(auth_url=auth["auth_url"], token=token, project_id=auth["project_id"])
        session = keystone_session.Session(auth=login, verify=settings.get("cacert") or settings.get("verify", True))
        conn = openstack.connection.Connection(session=session, region_name=settings.get("region_name"),
                                               interface=settings.get("interface"))
        try:
            conn.session.get_token()  # one request; fails if the token was revoked
            return conn
        except (auth_exceptions.Unauthorized, auth_exceptions.NotFound):
            pass
    conn = openstack.connection.Connection(config=config.get_one(cloud=CLOUD))
    save_token(auth, conn.session.auth.get_access(conn.session))
    return conn


def cached_token(auth, minutes):
    """Return the saved token of this user and project if it is valid for at least `minutes`, else None."""
    try:
        cache = json.loads(TOKEN_CACHE.read_text())
        left = datetime.datetime.fromisoformat(cache["expires_at"]) - datetime.datetime.now(datetime.timezone.utc)
    except (OSError, ValueError, KeyError, TypeError):  # missing or damaged file
        return None
    keys = ("auth_url", "username", "project_id")
    if not auth.get("project_id") or [cache.get(k) for k in keys] != [auth.get(k) for k in keys]:
        return None
    return cache.get("token") if left >= datetime.timedelta(minutes=minutes) else None


def save_token(auth, access):
    """Save the token of this login (valid for one hour on this cloud) for the next runs, readable only by you."""
    if not auth.get("project_id"):
        return
    data = {"auth_url": auth["auth_url"], "username": auth.get("username"), "project_id": auth["project_id"],
            "token": access.auth_token, "expires_at": access.expires.isoformat()}
    try:
        TOKEN_CACHE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=TOKEN_CACHE.parent, prefix="token.")  # mode 0600
        with os.fdopen(descriptor, "w") as f:
            json.dump(data, f)
        os.replace(temporary, TOKEN_CACHE)
    except OSError as e:
        print(f"Warning: cannot save the login token ({e}); the password will be asked for again.")


def logout():
    if TOKEN_CACHE.exists():
        TOKEN_CACHE.unlink()
        print("Deleted the saved login token; the next command asks for the password again.")
    else:
        print("No saved login token.")


def floating_ip(server):
    """Return the floating IPv4 address of a server, or a fixed one if it has none."""
    fixed = None
    for addresses in (server.addresses or {}).values():
        for address in addresses:
            if address.get("version") != 4:
                continue
            if address.get("OS-EXT-IPS:type") == "floating":
                return address["addr"]
            fixed = fixed or address["addr"]
    return fixed


def find_server(conn, name, servers=None):
    """Return your instance with this name or ID; instances of other users in the project are ignored."""
    me = conn.current_user_id
    matches = [s for s in (servers or conn.compute.servers()) if s.user_id == me and name in (s.name, s.id)]
    if not matches:
        sys.exit(f"You have no instance with the name or ID {name}.")
    if len(matches) > 1:
        sys.exit(f"You have several instances named {name}; pass one of these IDs instead: "
                 + ", ".join(s.id for s in matches))
    return matches[0]


def server_key(server):
    """Private key file the server was started with, or None to let ssh use the default keys or agent."""
    key = (server.metadata or {}).get("gpuaas_ssh_key")
    if key and not Path(key).expanduser().exists():
        print(f"Note: {key}, the key {server.name} was started with, is not on this computer; "
              "ssh tries your default keys.", file=sys.stderr)
        return None
    return key


def ssh_argv(key, ip, options=()):
    argv = ["ssh", "-o", "StrictHostKeyChecking=accept-new", *options]
    if key:
        argv += ["-i", str(Path(key).expanduser()), "-o", "IdentitiesOnly=yes"]
    return argv + [f"{SSH_USER}@{ip}"]


def remote(key, ip, command):
    """Run a command over SSH without any prompt; return (exit code, stdout or stderr)."""
    argv = ssh_argv(key, ip, ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]) + [command]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return 255, "timeout"
    return result.returncode, (result.stdout if result.returncode == 0 else result.stderr).strip()


def read_phase(key, ip):
    """(exit code, setup phase) of an instance, read over SSH."""
    return remote(key, ip, f"cat {PHASE_FILE}")


def forget_host_key(ip):
    """Floating IPs are reused, so known_hosts may still hold the key of an old instance."""
    try:
        subprocess.run(["ssh-keygen", "-R", ip], capture_output=True)
    except OSError:
        pass


def wait_for_ssh(ip, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((ip, 22), timeout=5):
                return True
        except OSError:
            time.sleep(5)
    return False


def tilde(path):
    path = Path(path)
    return f"~/{path.relative_to(Path.home())}" if path.is_relative_to(Path.home()) else str(path)


# ---------------------------------------------------------------- software packages

def load_packages():
    """Read cloud-init/packages/*.yaml; the `gpuaas:` block of each file describes the package (see docs/guide.md)."""
    catalog = {}
    for path in PACKAGE_DIR.glob("*.yaml"):
        try:
            config = yaml.safe_load(path.read_text())
            meta = config.pop("gpuaas")
            catalog[path.stem] = {"title": meta["title"], "group": meta.get("group", "Other"), "order": meta["order"],
                                  "required": meta.get("required", False), "requires": meta.get("requires", []),
                                  "os": meta.get("os", {}), "long": meta.get("long", False),
                                  "build": meta.get("build"), "check": meta.get("check"), "job": meta.get("job"),
                                  "disk": meta.get("disk", 0), "config": config}
        except KeyError as e:
            sys.exit(f"{path}: the package file lacks {e} (see docs/guide.md, 'Adding a package').")
        except (yaml.YAMLError, AttributeError, TypeError) as e:
            sys.exit(f"{path}: not a valid package file: {e}")
    catalog = dict(sorted(catalog.items(), key=lambda item: item[1]["order"]))
    for name, package in catalog.items():
        for dependency in package["requires"]:
            if dependency not in catalog or catalog[dependency]["order"] >= package["order"]:
                sys.exit(f"{PACKAGE_DIR / name}.yaml: dependency {dependency} must exist and have a lower order.")
    return catalog


def support(package, os_name):
    """Compatibility of a package with an OS: 'tested', 'untested' or 'no: <reason>' (unknown OS: 'untested')."""
    return str(package["os"].get(os_name, "untested" if os_name is None else "no: not in its compatibility list"))


def supported_os(catalog):
    return sorted({os_name for package in catalog.values() for os_name in package["os"]}, reverse=True)


def image_os(image):
    """OS of an image as 'ubuntu-24.04', from gpuaas, os_distro/os_version or the image name; None if unknown."""
    properties = image.properties or {}
    if properties.get("gpuaas_os"):
        return properties["gpuaas_os"]
    distro = getattr(image, "os_distro", None) or properties.get("os_distro")
    version = getattr(image, "os_version", None) or properties.get("os_version")
    if distro and version:
        return f"{distro}-{version}".lower()
    match = re.search(r"ubuntu[-_ ]?(\d{2}\.\d{2})", image.name or "", re.IGNORECASE)
    return f"ubuntu-{match.group(1)}" if match else None


def resolve_packages(catalog, wanted):
    """Return the required packages plus `wanted` and their dependencies, in install order."""
    unknown = set(wanted) - set(catalog)
    if unknown:
        sys.exit(f"Unknown package(s) {', '.join(sorted(unknown))}; available: {', '.join(catalog)}.")
    chosen = set()

    def add(name):
        chosen.add(name)
        for dependency in catalog[name]["requires"]:
            add(dependency)

    for name in [n for n, p in catalog.items() if p["required"]] + list(wanted):
        add(name)
    return [name for name in catalog if name in chosen]


def merge(target, source):
    """Merge a cloud-config fragment: lists (packages, runcmd, write_files) are appended in install order,
    dicts are merged, and different values of the same scalar are an error."""
    for key, value in source.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], list) and isinstance(value, list):
            target[key] = target[key] + copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            merge(target[key], value)
        elif target[key] != value:
            sys.exit(f"cloud-init fragments disagree on {key!r}.")


def finish_script(builds, checks, jobs):
    """Shell script run once by gpuaas-finish.service after the first boot (and the reboot):
    builds that need the loaded driver, then the checks, the ready marker and the jobs. Builds and jobs run
    with set -e in a subshell of their own, so that a failing line fails them, not only a failing last line.
    The checks are scripts in CHECK_DIR (`checks`: package and script, in install order), run in parallel:
    one per CPU core and 4 GB of memory, so sequentially on small and medium flavors."""
    lines = ["#!/bin/sh",
             "# Generated by gpuaas.py: builds and checks the software, marks the instance ready, runs the jobs.",
             f'phase() {{ echo "$1" > {PHASE_FILE}; }}',
             'finish() { systemctl disable gpuaas-finish.service; exit "$1"; }']
    for name, build in builds:
        lines += [f'phase "building {name}"', f'echo "== build {name}"', "(", "set -e", build.rstrip(), ")",
                  f'[ $? -eq 0 ] || {{ phase "failed: {name}"; finish 1; }}']
    lines += ["phase finishing",
              "jobs=$(awk '/^MemTotal/ {print int(($2 + 1048576) / 4194304)}' /proc/meminfo)",
              '[ "$jobs" -gt "$(nproc)" ] && jobs=$(nproc)',
              '[ "$jobs" -ge 1 ] || jobs=1',
              "mkdir -p /var/log/gpuaas && rm -f /var/log/gpuaas/check-*",
              # Each check's output goes to its own log, which is printed as one block when the check ends.
              # Named explicitly (slow ones first): an image may still hold scripts of its own start.
              f"printf '%s\\n' {' '.join(sorted(Path(script).name for _, script in checks))} | "
              f"xargs -r -n 1 -P \"$jobs\" sh -c 'log=/var/log/gpuaas/check-$1; "
              f"{CHECK_DIR}/$1 > \"$log.log\" 2>&1; echo $? > \"$log.status\"; "
              "flock /var/log/gpuaas/print.lock sh -c \"echo == check ${1#*-}; cat $log.log\"' check"]
    for name, script in checks:  # the first failure in install order
        status = f"/var/log/gpuaas/check-{Path(script).name}.status"
        lines.append(f'[ "$(cat {status} 2>/dev/null)" = 0 ] || {{ phase "failed: {name}"; finish 1; }}')
    lines.append("phase ready")
    for name, job in jobs:
        lines += [f'echo "== job {name}"', "(", "set -e", job.rstrip(), ")"]
    lines.append("finish 0")
    return "\n".join(lines) + "\n"


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, lambda dumper, value: dumper.represent_scalar(
    "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None))


def workspace_archive(names):
    """Base64 tar.xz of the files at the top of workspace/ (README.md, run_all.sh, plot_results.gp) and the
    workspace folders of the given packages. xz instead of gzip, because its large dictionary also removes
    the repetitions between folders (common.h and similar programs); gzip's 32 KiB window does not, and with
    all packages the user data then exceeded Nova's limit even compressed."""
    def clean(info):
        if Path(info.name).name in ("build", ".DS_Store", "__pycache__"):
            return None
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        if info.isfile():
            info.mode = 0o755 if info.name.endswith(".sh") else 0o644
        return info

    buffer = io.BytesIO()
    # Preset 6 has an 8 MiB dictionary, more than the workspace; higher presets only use more memory.
    with tarfile.open(fileobj=buffer, mode="w:xz", preset=6) as tar:
        for file in sorted(f for f in WORKSPACE_DIR.iterdir() if f.is_file()):
            tar.add(file, arcname=file.name, filter=clean)
        for name in names:
            tar.add(WORKSPACE_DIR / name, arcname=name, filter=clean)
    return base64.b64encode(buffer.getvalue()).decode()


def build_userdata(catalog, packages, preinstalled, burn_args):
    """Merge the package fragments into one cloud-config. Packages already in the image get no install or build,
    only their check, job and workspace folder. A package without its own check runs its workspace run.sh."""
    config, builds, checks, jobs = {}, [], [], []
    folders = [name for name in packages if (WORKSPACE_DIR / name).is_dir()]
    for name in packages:
        package = catalog[name]
        if name not in preinstalled:
            merge(config, package["config"])
            if package["build"]:
                builds.append((name, package["build"]))
        check = package["check"] or (name in folders and f"gpuaas-check-workspace {name}")
        if check:
            checks.append((name, check))
        if package["job"]:
            jobs.append((name, package["job"].replace("@BURN_ARGS@", burn_args)))
    # The slow checks (the HPC packages, which come last in install order) start first.
    first = sorted(checks, key=lambda item: -catalog[item[0]]["order"])
    scripts = {name: f"{CHECK_DIR}/{number:02d}-{name}" for number, (name, _) in enumerate(first, 1)}
    files = [{"path": PHASE_FILE, "content": "installing\n"},
             {"path": "/usr/local/sbin/gpuaas-finish", "permissions": "0755",
              "content": finish_script(builds, [(name, scripts[name]) for name, _ in checks], jobs)},
             *({"path": scripts[name], "permissions": "0755", "content": f"#!/bin/sh\nset -e\n{check.rstrip()}\n"}
               for name, check in checks),
             {"path": "/etc/systemd/system/gpuaas-finish.service", "content": FINISH_UNIT},
             {"path": "/usr/local/sbin/gpuaas-check-workspace", "permissions": "0755", "content": CHECK_WORKSPACE_SCRIPT},
             {"path": WORKSPACE_TAR, "encoding": "b64", "content": workspace_archive(folders)}]
    merge(config, {
        "package_update": True,
        "package_upgrade": True,
        "packages": ["xz-utils"],  # to unpack the workspace
        "write_files": files,
        "runcmd": [
            # A kernel update from package_upgrade needs the same single reboot as the driver.
            "test -f /var/run/reboot-required && touch /run/gpuaas-reboot",
            # Existing files are kept, so changes survive snapshots; new folders are added.
            f"mkdir -p {REMOTE_WORKSPACE} && tar -xJf {WORKSPACE_TAR} -C {REMOTE_WORKSPACE} "
            f"--skip-old-files && chown -R {SSH_USER}:{SSH_USER} {REMOTE_WORKSPACE}",
            # The checks run copies of this unchanged version; the one in the home directory may have been edited.
            f"rm -rf {CHECK_WORKSPACE} && mkdir -p {CHECK_WORKSPACE} && tar -xJf {WORKSPACE_TAR} -C {CHECK_WORKSPACE}",
            f"test -f /run/gpuaas-reboot && echo rebooting > {PHASE_FILE}",
            "systemctl daemon-reload",
            "systemctl enable gpuaas-finish.service",
            "test -f /run/gpuaas-reboot || systemctl start --no-block gpuaas-finish.service",
        ],
        "power_state": {"mode": "reboot", "message": "Rebooting to load the NVIDIA driver",
                        "condition": "test -f /run/gpuaas-reboot"},
    })
    return "#cloud-config\n" + yaml.dump(config, Dumper=_Dumper, sort_keys=False, width=1000)


def fit_userdata(data):
    """Nova accepts at most 64 KiB of base64-encoded user data; gzip it if needed (cloud-init unpacks it)."""
    if len(base64.b64encode(data.encode())) <= USERDATA_LIMIT:
        return data
    packed = gzip.compress(data.encode())
    if len(base64.b64encode(packed)) > USERDATA_LIMIT:
        sys.exit("The cloud-init data is too large even when compressed; choose fewer packages or shrink workspace/.")
    return packed


# ---------------------------------------------------------------- wizard

def ask(question):
    answer = question.ask()
    if answer is None:  # Ctrl-C
        sys.exit("Aborted.")
    return answer


def free_name(base, taken):
    name, number = base, 2
    while name in taken:
        name, number = f"{base}-{number}", number + 1
    return name


def gpu_flavors(conn):
    """Flavors with a GPU (pci_passthrough:alias such as '3080:2'), as (flavor, '2x RTX 3080', supported)."""
    flavors = []
    for flavor in conn.compute.flavors():
        alias = (flavor.extra_specs or {}).get("pci_passthrough:alias")
        model, _, count = (alias or "").partition(":")
        if model and (count or "1").isdigit():  # skips flavors without a GPU and aliases this tool does not know
            flavors.append((flavor, model, int(count or 1)))
    flavors.sort(key=lambda f: (f[1], f[2], f[0].vcpus, f[0].ram))
    return [(flavor, f"{count}x RTX {model.replace('ti', ' Ti')}", model in SUPPORTED_GPUS)
            for flavor, model, count in flavors]


def flavor_title(flavor, gpus):
    return f"{flavor.name:<18} {gpus:<14} {flavor.vcpus:>2} vCPU {flavor.ram // 1024:>3} GB RAM {flavor.disk:>4} GB disk"


def image_packages(image):
    return set(filter(None, ((image.properties or {}).get("gpuaas_packages") or "").split(",")))


def gpu_images(conn, os_names):
    """Public images of the supported OS versions, your snapshots, and shared or community gpuaas images."""
    found = {}
    for image in [*conn.image.images(), *conn.image.images(visibility="community")]:
        properties = image.properties or {}
        own_snapshot = image.owner == conn.current_project_id and properties.get("image_type") == "snapshot"
        public_os = image.visibility == "public" and image_os(image) in os_names
        if image.status == "active" and (public_os or own_snapshot or image_packages(image)):
            found[image.id] = image
    return sorted(found.values(), key=lambda image: (image.name != DEFAULT_IMAGE, image.name or ""))


def image_label(image, project_id):
    """Menu name of an image: its name, plus who provides it unless it is public."""
    if image.visibility == "public":
        return image.name
    if image.owner == project_id:
        return f"{image.name} (your project)"
    return f"{image.name} ({image.visibility}, project {(image.owner or '?')[:8]})"


def pretty_os(os_name):
    return os_name.replace("ubuntu-", "Ubuntu ") if os_name else "unknown OS"


def os_hint(package, catalog):
    """' (Ubuntu 24.04 only)' if a package does not run on every OS that the packages support."""
    all_os = supported_os(catalog)
    fine = [os_name for os_name in all_os if not support(package, os_name).startswith("no")]
    return f" ({', '.join(pretty_os(os_name) for os_name in fine)} only)" if len(fine) < len(all_os) else ""


def rate_image(image, catalog, packages):
    """Packages that cannot run on the image's OS, and packages untested there."""
    levels = {name: support(catalog[name], image_os(image)) for name in packages}
    return ([name for name, level in levels.items() if level.startswith("no")],
            [name for name, level in levels.items() if level == "untested"])


def best_image(images, rated):
    """ID of the public image for the chosen packages: fewest untested ones, the default image first, then the newest."""
    fitting = [i for i, image in images.items() if not rated[i][0] and image.visibility == "public"]
    fitting.sort(key=lambda i: images[i].updated_at or "", reverse=True)
    fitting.sort(key=lambda i: (len(rated[i][1]), images[i].name != DEFAULT_IMAGE))
    return fitting[0] if fitting else None


def find_image(conn, images, name):
    """ID of the image given by --image or "Other image": an ID, or a name (public images first)."""
    if name in images:
        return name
    matches = [i for i, image in images.items() if image.name == name]
    public = [i for i in matches if images[i].visibility == "public"]
    if len(public) == 1 or len(matches) == 1:
        return (public or matches)[0]
    if matches:
        sys.exit(f"Several images are named {name}; give the ID of one: " + ", ".join(matches))
    try:
        image = conn.image.find_image(name)
    except exceptions.DuplicateResource:
        sys.exit(f"Several images are named {name}; give its ID instead.")
    if image is None:
        sys.exit(f"Image {name} not found.")
    images[image.id] = image
    return image.id


def ssh_public_keys():
    keys = sorted(p for p in (Path.home() / ".ssh").glob("*.pub") if not p.name.endswith("-cert.pub"))
    preferred = [p for name in ("id_ed25519.pub", "id_rsa.pub") for p in keys if p.name == name]
    return preferred + [p for p in keys if p not in preferred]


def key_title(path):
    fields = path.read_text().split() or ["empty file"]
    return f"{tilde(path)}  ({fields[0]}{', ' + fields[2] if len(fields) > 2 else ''})"


def choose_key(args, ask_user):
    """Return ('pub', path) or ('new', path)."""
    keys = ssh_public_keys()
    if args.key == "new":
        default = ("pub", NEW_KEY_FILE.with_suffix(".pub").resolve()) if NEW_KEY_FILE.exists() else ("new", NEW_KEY_FILE)
    elif args.key:
        path = Path(args.key).expanduser()
        if not path.is_file() or path.suffix != ".pub":
            sys.exit(f"--key {args.key}: give a public key file (.pub) or 'new'.")
        default = ("pub", path.resolve())
    else:
        default = ("pub", keys[0].resolve()) if keys else ("new", NEW_KEY_FILE)
    if not ask_user:
        return default
    choices = [questionary.Choice(key_title(p), value=("pub", p.resolve())) for p in keys]
    if not NEW_KEY_FILE.exists():
        choices.append(questionary.Choice(f"Generate a new key ({tilde(NEW_KEY_FILE)})", value=("new", NEW_KEY_FILE)))
    if default not in [c.value for c in choices]:
        choices.insert(0, questionary.Choice(key_title(default[1]) if default[0] == "pub" else str(default[1]),
                                             value=default))
    return ask(questionary.select("SSH key:", choices=choices, default=default))


def ensure_keypair(conn, key_choice, ask_user):
    """Make sure the chosen key exists in OpenStack; return (keypair name, private key file or None)."""
    kind, path = key_choice
    if kind == "new":
        path.parent.mkdir(mode=0o700, exist_ok=True)
        # ssh-keygen asks for a passphrase in a terminal; without one the key gets none.
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-f", str(path), "-C", "gpuaas"]
                       + ([] if ask_user else ["-N", ""]), check=True)
        path = path.with_name(path.name + ".pub")
    public_key = path.read_text().strip()
    if not public_key.startswith(("ssh-", "ecdsa-", "sk-")):
        sys.exit(f"{path} is not an OpenSSH public key.")
    # The name contains a hash of the key, so the same key always maps to the same keypair;
    # older Nova API versions allow only letters, digits, "-" and "_" in keypair names.
    digest = hashlib.sha256(public_key.split()[1].encode()).hexdigest()[:8]
    name = f"gpuaas-{re.sub(r'[^A-Za-z0-9_-]', '-', path.stem)}-{digest}"
    if conn.compute.find_keypair(name) is None:
        conn.compute.create_keypair(name=name, public_key=public_key)
        print(f"Imported {tilde(path)} as keypair {name}.")
    private = path.with_suffix("")
    return name, (tilde(private) if private.exists() else None)


# ---------------------------------------------------------------- commands

def cmd_start(conn, args):
    catalog = args.catalog
    custom_userdata = Path(args.userdata).read_text() if args.userdata else None
    ask_user = interactive(args)
    # From the token, so it works for password and saved-token logins alike (only the first has conn.config.auth).
    username = conn.session.auth.get_access(conn.session).username or getpass.getuser()

    # Name
    taken = {s.name for s in conn.compute.servers()}
    if args.name in taken and not ask_user:
        sys.exit(f"An instance named {args.name} already exists in the project; use another name.")
    name = args.name if args.name and args.name not in taken else free_name(args.name or f"{username}-gpu", taken)
    if ask_user:
        name = ask(questionary.text("Instance name:", default=name, validate=lambda value: (
            "Enter a name." if not value.strip() else
            "This name is already used in the project." if value.strip() in taken else True))).strip()

    # Flavor
    flavors = gpu_flavors(conn)
    flavor = args.flavor or DEFAULT_FLAVOR
    titles = {f.name: flavor_title(f, gpus) for f, gpus, _ in flavors}
    flavor_objects = {f.name: f for f, _, _ in flavors}
    unsupported = {f.name for f, _, supported in flavors if not supported}
    if flavor in unsupported:
        sys.exit(f"{flavor} is not supported (only RTX {' and '.join(SUPPORTED_GPUS)}); the default is {DEFAULT_FLAVOR}.")
    if ask_user:
        choices = [questionary.Choice(title, value=flavor_name,
                                      disabled="not supported" if flavor_name in unsupported else None)
                   for flavor_name, title in titles.items()]
        if flavor not in titles:
            choices.insert(0, questionary.Choice(f"{flavor} (no GPU flavor)", value=flavor))
        flavor = ask(questionary.select("Flavor:", choices=choices, default=flavor))

    # Software, asked before the image so that the image can be chosen to fit it; an image given with --image
    # shows what it already has (added after the image question anyway).
    images = {image.id: image for image in gpu_images(conn, supported_os(catalog))}
    given = find_image(conn, images, args.image) if args.image else None
    in_given = {p for p in image_packages(images[given]) & set(catalog) if not catalog[p]["job"]} if given else set()
    packages, wanted, burn_args = [], set(), ""
    if custom_userdata is None:
        wanted = set(filter(None, (name.strip() for name in (args.packages or "").split(","))))
        if args.burn is not None:
            wanted.add("gpu-burn")
        resolve_packages(catalog, wanted)  # report unknown names before asking
        if ask_user:
            choices, group = [], None
            for package_name, package in catalog.items():
                if package["group"] != group:
                    group = package["group"]
                    choices.append(questionary.Separator(f"-- {group}"))
                title = package["title"] + os_hint(package, catalog)
                if package["required"]:
                    choices.append(questionary.Choice(title, value=package_name, checked=True, disabled="always"))
                elif package_name in in_given:
                    choices.append(questionary.Choice(title, value=package_name, checked=True, disabled="in the image"))
                else:
                    choices.append(questionary.Choice(title, value=package_name, checked=package_name in wanted))
            wanted = set(ask(questionary.checkbox("Software (space selects, enter confirms):", choices=choices)))
        packages = resolve_packages(catalog, wanted)
        added = [p for p in packages if p not in wanted and not catalog[p]["required"]]
        if added:
            print(f"Added as dependency: {', '.join(added)}")

    # Image: images whose OS does not support all chosen packages are disabled; the best one is preselected.
    # Keyed by ID, because names need not be unique; images that are not public show who provides them.
    rated = {i: rate_image(image, catalog, packages) for i, image in images.items()}
    best = best_image(images, rated)
    image = given or best
    if image is None:
        sys.exit("No image supports all of: " + ", ".join(packages) + "; see 'gpuaas.py help start'.")
    if ask_user:
        choices = []
        for image_id, candidate in images.items():
            problems, untested = rated[image_id]
            has = sorted(image_packages(candidate))
            title = (f"{image_label(candidate, conn.current_project_id)}  [{pretty_os(image_os(candidate))}]"
                     + (f"  has: {', '.join(has)}" if has else "")
                     + (f"  untested: {', '.join(untested)}" if untested else ""))
            choices.append(questionary.Choice(title, value=image_id,
                                              disabled=f"not for {', '.join(problems)}" if problems else None))
        if rated[image][0] and best:
            image = best
        choices.append(questionary.Choice("Other image (enter its name or ID)", value=OTHER))
        image = ask(questionary.select("Image:", choices=choices, default=image))
        if image == OTHER:
            image = find_image(conn, images, ask(questionary.text(
                "Image name or ID:", validate=lambda v: bool(v.strip()) or "Enter a name.")).strip())
    image_object = images[image]
    image_name = image_label(image_object, conn.current_project_id)
    os_name = image_os(image_object)
    problems, untested = rate_image(image_object, catalog, packages)
    if problems:
        sys.exit(f"{image_name} ({pretty_os(os_name)}) does not support: "
                 + "; ".join(f"{p} ({support(catalog[p], os_name)[3:].strip()})" for p in problems))
    if untested:
        print(f"Note: untested on {pretty_os(os_name)}: {', '.join(untested)}")
    preinstalled = image_packages(image_object) & set(catalog)
    if custom_userdata is None:
        # Packages in the image are checked too, except those with a job (gpu-burn would run again).
        packages = resolve_packages(catalog, wanted | {p for p in preinstalled if not catalog[p]["job"]})
        if "gpu-burn" in packages:
            seconds = args.burn or DEFAULT_BURN_SECONDS
            if ask_user:
                seconds = int(ask(questionary.text("gpu-burn duration in seconds:", default=str(seconds),
                                                   validate=lambda v: v.isdigit() and int(v) > 0 or "Enter seconds.")))
            burn_args = shlex.join([*shlex.split(args.burn_args), str(seconds)])  # checked in main()

    # SSH key and network (NetLab sets up a network <username>-net for each account)
    key_choice = choose_key(args, ask_user)
    network = args.network or f"{username}-net"
    if conn.network.find_network(network) is None:
        names = sorted(n.name for n in conn.network.networks())
        sys.exit(f"There is no network {network}; choose one with --network: {', '.join(names) or 'none found'}.")

    kind, key_path = key_choice
    print(f"\n  name      {name}\n  flavor    {titles.get(flavor, flavor)}\n  image     {image_name} ({pretty_os(os_name)})")
    print(f"  software  {', '.join(packages) if packages else 'own cloud-init file ' + args.userdata}")
    print(f"  ssh key   {'new key ' if kind == 'new' else ''}{tilde(key_path)}\n  network   {network}")
    disk = flavor_objects[flavor].disk if flavor in flavor_objects else None
    need = BASE_DISK_GB + sum(catalog[p]["disk"] for p in set(packages) | preinstalled)
    if disk and need > disk - 2:  # the file system has about 2 GB less than the flavor's disk
        print(f"Warning: the software needs about {need:.0f} GB of disk, but {flavor} has {disk} GB.")
    source_builds = [p for p in packages if catalog[p]["long"] and p not in preinstalled]
    ready_timeout = args.ready_timeout or (12 * 60 if source_builds else 60)
    if source_builds:
        print(f"Note: {', '.join(source_builds)} {'comes' if len(source_builds) == 1 else 'come'} from the GPUaaS "
              "binary cache after the reboot (about 15 min); whatever the cache lacks is built from source, "
              "which can take hours.")
        if flavor in flavor_objects and flavor_objects[flavor].ram < 16 * 1024:
            print("Warning: if something has to be built from source, use an xlarge flavor (16 GB RAM, 160 GB disk).")
    data = fit_userdata(custom_userdata if custom_userdata is not None
                        else build_userdata(catalog, packages, preinstalled, burn_args))
    if ask_user and not ask(questionary.confirm("Start this instance?", default=True)):
        sys.exit("Aborted.")

    keypair, private_key = ensure_keypair(conn, key_choice, ask_user)
    meta = {"gpuaas_packages": ",".join(sorted(set(packages) | preinstalled)), "gpuaas_os": os_name or ""}
    if private_key:
        meta["gpuaas_ssh_key"] = private_key
    print(f"Starting {name} ...", flush=True)
    try:
        server = conn.create_server(name, image=image_object.id, flavor=flavor, network=network, key_name=keypair,
                                    userdata=data, meta=meta, auto_ip=True, wait=True, timeout=args.timeout)
    except exceptions.SDKException as e:
        message = f"Start failed: {e}"
        if "No valid host" in str(e):
            message += "\nUsually this means that no GPU of this flavor is free right now."
        failed = conn.get_server(name)
        if failed is not None and failed.status == "ERROR":
            try:
                conn.delete_server(failed.id, wait=True, delete_ips=True)
                message += "\nThe failed instance was deleted."
            except exceptions.SDKException as delete_error:
                message += (f"\nDeleting the failed instance failed too ({delete_error}); "
                            f"retry with 'gpuaas.py delete {name}'.")
        elif failed is not None:
            message += f"\nThe instance still exists ({failed.status}); check 'gpuaas.py list' or delete it."
        sys.exit(message)

    ip = server.public_v4 or floating_ip(server)
    if not ip:
        sys.exit(f"Started {server.name} ({server.id}), but it has no IP address; check 'gpuaas.py list'.")
    print(f"Started {server.name} ({server.id}), IP {ip}.")
    forget_host_key(ip)
    print(f"Login with: {shlex.join(ssh_argv(private_key, ip))}\n        or: gpuaas.py ssh {server.name}")
    if args.no_wait:
        return
    if custom_userdata is not None:
        print("Waiting for SSH ...", flush=True)
        if not wait_for_ssh(ip, SSH_WAIT_SECONDS):
            print(f"SSH did not answer within {SSH_WAIT_SECONDS} s; the instance may still be booting.")
        return
    if not wait_until_ready(server.name, private_key, ip, ready_timeout):
        sys.exit(1)
    if "gpu-burn" in packages:
        print(f"gpu-burn runs now; results: {HOME_DIR}/gpu-burn/gpu-burn-results_*.log")


def wait_until_ready(name, key, ip, minutes):
    """Follow the setup phase over SSH until the instance is ready or failed."""
    print(f"Waiting until the software is installed (up to {minutes} min; Ctrl-C stops waiting, "
          "the instance keeps installing) ...", flush=True)
    deadline = time.monotonic() + minutes * 60
    last, failures = None, 0
    try:
        while time.monotonic() < deadline:
            code, output = read_phase(key, ip)
            if code != 0 and "Permission denied" in output:
                print("SSH login without a prompt failed (key with passphrase? add it to ssh-agent); stopped waiting.")
                return False
            if code != 0 and ("Host key verification failed" in output or "Too many authentication failures" in output):
                print(f"SSH failed: {output}\nStopped waiting; fix this (e.g. remove the old host key of {ip} from "
                      "your known_hosts files) and follow the installation with: gpuaas.py logs " + name)
                return False
            failures = 0 if code == 0 else failures + 1
            # A single failed poll during the installation (busy instance, sshd restarted by an upgrade) keeps the
            # last phase; only a reboot or a longer outage shows as waiting.
            if code != 0 and last not in (None, "rebooting") and failures < 3:
                time.sleep(READY_POLL_SECONDS)
                continue
            phase = output if code == 0 else "waiting for SSH (booting or rebooting)"
            if phase != last:
                print(f"  {time.strftime('%H:%M:%S')}  {phase}", flush=True)
                last = phase
            if phase == "ready":
                print(remote(key, ip, "nvidia-smi")[1])
                return True
            if phase.startswith("failed"):
                print(f"Setup {phase}; see the log with: gpuaas.py logs {name}")
                return False
            time.sleep(READY_POLL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nStopped waiting; follow the installation with: gpuaas.py logs {name}")
        return False
    print(f"Not ready after {minutes} min; check with: gpuaas.py logs {name}")
    return False


def shelve_in(server):
    if server.status != "ACTIVE" or not server.launched_at:
        return "-"
    launched = datetime.datetime.fromisoformat(server.launched_at).replace(tzinfo=datetime.timezone.utc)
    left = launched + datetime.timedelta(days=SHELVE_AFTER_DAYS) - datetime.datetime.now(datetime.timezone.utc)
    return f"~{left.days} d" if left.days >= 1 else "<1 d"


def cmd_list(conn, args):
    me = conn.current_user_id
    servers = sorted((s for s in conn.compute.servers() if args.all or s.user_id == me),
                     key=lambda s: s.created_at or "")
    if servers:
        rows = [["NAME", "STATUS", "FLAVOR", "IP", "CREATED", "SHELVED IN"] + (["USER"] if args.all else [])]
        for s in servers:
            rows.append([s.name, s.status, (s.flavor.name if s.flavor else None) or "-", floating_ip(s) or "-",
                         (s.created_at or "")[:16].replace("T", " "), shelve_in(s)]
                        + ([s.user_id[:8]] if args.all else []))
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        for row in rows:
            print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    else:
        print("No instances.")
    limits = conn.get_compute_limits()
    print(f"\nProject quota: instances {limits.instances_used}/{limits.instances}, "
          f"vCPUs {limits.total_cores_used}/{limits.total_cores}, "
          f"RAM {limits.total_ram_used // 1024}/{limits.total_ram // 1024} GB")


def running_server(conn, args):
    """Your instance by name; a shelved one is unshelved first (after asking in a terminal)."""
    server = find_server(conn, args.name)
    if server.status in SHELVED:
        if not (interactive() and ask(questionary.confirm(
                f"{server.name} is shelved. Unshelve it now (needs a free GPU of {server.flavor.name})?", default=True))):
            sys.exit(f"{server.name} is shelved; resume it with: gpuaas.py unshelve {server.name}")
        # Unshelving can take minutes, so the login token must stay valid that long.
        conn = connect(TOKEN_MINUTES["unshelve"])
        server = unshelve(conn, server)
    elif server.status != "ACTIVE":
        sys.exit(f"{server.name} is {server.status}, not running.")
    return server


def exec_ssh(conn, args, options=(), command=()):
    """Replace gpuaas.py with ssh to your running instance (a shelved one is unshelved first)."""
    server = running_server(conn, args)
    ip = floating_ip(server)
    if not ip:
        sys.exit(f"{server.name} has no IP address.")
    os.execvp("ssh", ssh_argv(server_key(server), ip, options) + list(command))


def cmd_ssh(conn, args):
    exec_ssh(conn, args, command=args.command)


def cmd_logs(conn, args):
    exec_ssh(conn, args, ["-t"], ["sudo journalctl -f -n 80 -u cloud-final.service -u gpuaas-finish.service"])


def cmd_snapshot(conn, args):
    server = find_server(conn, args.name)
    if conn.image.find_image(args.image) is not None:
        sys.exit(f"An image named {args.image} already exists.")
    # The image lists the packages chosen at start; only a ready instance has them all working.
    ip = floating_ip(server) if server.status == "ACTIVE" else None
    key = server_key(server)
    code, phase = read_phase(key, ip) if ip else (255, "")
    if code == 0 and phase != "ready":
        if not args.force:
            sys.exit(f"{server.name} is not ready ({phase}), so the image would list software that does not work; "
                     "use --force to snapshot anyway.")
        # Otherwise instances started from the image would run this instance's unfinished setup at boot.
        remote(key, ip, "sudo systemctl disable gpuaas-finish.service")
    if code != 0:
        if args.share:
            sys.exit(f"--share needs SSH access to {server.name} to clean it up; start it or check the key.")
        print("Could not check over SSH whether the instance is ready; snapshotting anyway.")
    if args.share:
        # A snapshot keeps everything, and cloud-init only adds the key of each new instance to authorized_keys.
        code, output = remote(key, ip, f"sudo sh -c 'rm -f {HOME_DIR}/.ssh/authorized_keys {HOME_DIR}/.*_history "
                                       f"{HOME_DIR}/.julia/logs/repl_history.jl /root/.*_history && cloud-init clean --logs'")
        if code != 0:
            sys.exit(f"Cleaning up {server.name} for sharing failed: {output}")
        print(f"Removed the SSH keys, shell histories and cloud-init state of {server.name}; "
              "you cannot log in to it any more.")
    print(f"Creating image {args.image} from {server.name}; this can take several minutes ...", flush=True)
    metadata = server.metadata or {}
    image = conn.create_image_snapshot(args.image, server, wait=True, timeout=WAIT_MINUTES["snapshot"] * 60,
                                       gpuaas_packages=metadata.get("gpuaas_packages", ""),
                                       gpuaas_os=metadata.get("gpuaas_os", ""))
    print(f"Image {image.name} is {image.status}. It appears in the image menu of 'gpuaas.py start'.")
    if args.share:
        print(f"Make it visible to others with 'openstack image set --community {image.name}' (if the cloud allows it), "
              f"and delete the instance: gpuaas.py delete {server.name}")


def wait_for_task(conn, server, done, minutes):
    """Poll until the server reached a status in `done`, or its task (shelving, unshelving) ended; return it.
    A failed unshelve ends without a task in its old status, so the end of the task counts too."""
    deadline, busy = time.monotonic() + minutes * 60, False
    while time.monotonic() < deadline:
        server = conn.compute.get_server(server.id)
        busy = busy or bool(server.task_state)
        if server.status == "ERROR" or (not server.task_state and (busy or server.status in done)):
            return server
        time.sleep(5)
    sys.exit(f"{server.name} is still {server.status} ({server.task_state}) after {minutes} min; check 'gpuaas.py list'.")


def unshelve(conn, server):
    print(f"Unshelving {server.name}; this takes a few minutes ...", flush=True)
    conn.compute.unshelve_server(server)
    server = wait_for_task(conn, server, ("ACTIVE",), WAIT_MINUTES["unshelve"])
    if server.status != "ACTIVE":
        reason = (server.fault or {}).get("message", "")
        sys.exit(f"Unshelving {server.name} failed ({server.status}{': ' + reason if reason else ''})."
                 + (f"\nUsually this means that no GPU of {server.flavor.name} is free right now; try again later."
                    if server.status in SHELVED else ""))
    ip = floating_ip(server)
    print(f"{server.name} is running again, IP {ip}; waiting for SSH ...", flush=True)
    if ip and not wait_for_ssh(ip, SSH_WAIT_SECONDS):
        print(f"SSH did not answer within {SSH_WAIT_SECONDS} s; the instance may still be booting.")
    return server


def cmd_shelve(conn, args):
    server = find_server(conn, args.name)
    if server.status in SHELVED:
        sys.exit(f"{server.name} is already shelved.")
    print(f"Shelving {server.name}: its disk is saved as an image, then the GPU is released; "
          "this can take several minutes ...", flush=True)
    conn.compute.shelve_server(server)
    server = wait_for_task(conn, server, SHELVED, WAIT_MINUTES["shelve"])
    if server.status not in SHELVED:
        sys.exit(f"Shelving {server.name} failed ({server.status}).")
    print(f"{server.name} is shelved; 'gpuaas.py unshelve {server.name}' (or ssh) resumes it if a GPU is free.")


def cmd_unshelve(conn, args):
    server = find_server(conn, args.name)
    if server.status == "ACTIVE":
        sys.exit(f"{server.name} is running, not shelved.")
    if server.status not in SHELVED:
        sys.exit(f"{server.name} is {server.status}, not shelved.")
    server = unshelve(conn, server)
    print(f"Log in with: gpuaas.py ssh {server.name}")


def cmd_rename(conn, args):
    servers = list(conn.compute.servers())
    server = find_server(conn, args.name, servers)
    new = args.new_name.strip()
    if new in {s.name for s in servers}:
        sys.exit(f"An instance named {new} already exists in the project; use another name.")
    # The hostname as Nova would derive it: lowercase letters, digits and hyphens, at most 63 characters.
    hostname = re.sub(r"[^a-z0-9-]+", "-", new.lower()).strip("-")[:63].strip("-")
    if not hostname:
        sys.exit(f"{new} has no letters or digits for a hostname; use another name.")
    old_hostname = getattr(server, "hostname", None) or server.name
    # Nova API 2.90 can change the hostname too; cloud-init sets it from the metadata at every boot.
    conn.compute.put(f"/servers/{server.id}", json={"server": {"name": new, "hostname": hostname}},
                     microversion="2.90").raise_for_status()
    print(f"Renamed {server.name} to {new} (hostname {hostname}).")
    ip = floating_ip(server) if server.status == "ACTIVE" else None
    if not ip:
        print("It gets the new hostname when it runs again.")
        return
    # Both hostnames consist of [a-z0-9-] only, so they need no quoting in sed.
    hosts = (f" && sudo sed -i 's/\\b{old_hostname}\\b/{hostname}/g' /etc/hosts"
             if re.fullmatch(r"[a-z0-9-]+", old_hostname) else "")
    code, output = remote(server_key(server), ip, f"sudo hostnamectl set-hostname {hostname}{hosts}")
    print("The hostname is changed now; the prompt shows it at the next login." if code == 0 else
          f"Could not change the hostname over SSH ({output}); it changes at the next reboot.")


def cmd_delete(conn, args):
    server = find_server(conn, args.name)
    if interactive(args) and not ask(questionary.confirm(
            f"Delete {server.name} ({server.flavor.name if server.flavor else 'unknown flavor'}, "
            f"{floating_ip(server) or 'no IP'}) with its disk?", default=False)):
        sys.exit("Aborted.")
    conn.delete_server(server.id, wait=True, delete_ips=True)
    print(f"Deleted {server.name} and released its floating IP.")


EPILOG = """\
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
"""

COMMAND_HELP = {
    "start": "Start a GPU instance. In a terminal, start asks for the name, the flavor (live list with GPU\n"
             "model and count), the software, the image (only images whose Ubuntu version supports the\n"
             "software can be chosen, the best one is preselected), the gpu-burn duration and the SSH key,\n"
             "then shows a summary to confirm. Options preselect the answers; with --yes, or without a\n"
             "terminal, they are used without asking. The first boot installs all updates and the software;\n"
             "start waits until everything is installed and checked, then prints nvidia-smi. CPU and GPU\n"
             "examples for the software are in ~/workspace on the instance (from workspace/ in this repo).",
    "list": "List your instances with status, flavor, IP, creation time and the approximate days until\n"
            "automatic shelving (one week after launch), followed by the project quota. Shelved\n"
            "instances show the status SHELVED_OFFLOADED; unshelve resumes them.",
    "ssh": "Log in to one of your instances as user ubuntu, or run a command there, e.g.\n"
           "gpuaas.py ssh my-gpu nvidia-smi. ssh options after the name are passed on, e.g.\n"
           "gpuaas.py ssh my-gpu -L 8888:localhost:8888 to forward a port.",
    "logs": "Follow the installation and check output (cloud-init and gpuaas-finish) of one of your\n"
            "instances; Ctrl-C stops following.",
    "snapshot": "Save one of your instances as an image. The image remembers its software packages and\n"
                "appears in the image menu of start, which then only checks them instead of installing.\n"
                "Instances whose setup failed or is still running are refused unless --force is given.\n"
                "Before an image is shared with others, --share removes your SSH keys, shell histories and\n"
                "cloud-init state from the instance (you cannot log in to it afterwards).",
    "shelve": "Shelve one of your instances: its disk is saved as an image and the instance stops,\n"
              "which releases its GPU for others. Instances are shelved automatically one week after\n"
              "launch. Files and software are kept; unshelve resumes the instance.",
    "unshelve": "Resume a shelved instance with its files and software, if a GPU of its flavor is free\n"
                "(otherwise try again later). It keeps its IP address. ssh and logs offer this too when\n"
                "the instance is shelved.",
    "rename": "Rename one of your instances. Its hostname changes too: at once if it is running (the\n"
              "prompt shows it at the next login), otherwise when it runs again.",
    "delete": "Delete one of your instances with its disk and release its floating IP. In a terminal,\n"
              "delete asks first; --yes skips the question.",
    "setup": "Create clouds.yaml next to gpuaas.py from clouds.yaml.example: asks for your OpenStack\n"
             "username and password and lets you choose your project. The password is not stored.\n"
             "The first run of any command does this automatically if no clouds.yaml exists.",
    "logout": f"Delete the saved login token ({tilde(TOKEN_CACHE)}), e.g. on a shared computer. Every\n"
              "login saves its token, which this cloud accepts for one hour, so that the password is\n"
              "asked for only about once an hour.",
}


@contextlib.contextmanager
def friendly_errors():
    """Turn failed logins, an unreachable API and other cloud errors into one-line messages (also for tools/)."""
    try:
        yield
    except KeyboardInterrupt:
        sys.exit("\nAborted.")
    except auth_exceptions.Unauthorized:
        sys.exit("Error: login failed; check your username and password (gpuaas.py setup changes the username).")
    except auth_exceptions.SSLError as e:
        sys.exit(f"Error: {e}\nThe cloud's service endpoints use certificates of its own CA; clouds.yaml needs "
                 "verify: false (see clouds.yaml.example).")
    except (auth_exceptions.ConnectFailure, auth_exceptions.DiscoveryFailure) as e:
        sys.exit(f"Error: cannot reach the OpenStack API: {e}\n"
                 'Connect to the HS Fulda network or the VPN profile "05)_AI-NetLab-Pro".')
    except (exceptions.SDKException, auth_exceptions.ClientException, OSError, subprocess.CalledProcessError) as e:
        sys.exit(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true", help="log all OpenStack API calls")
    sub = parser.add_subparsers(dest="action", metavar="COMMAND")
    commands = {}

    def command(name, summary, epilog=None, instance=False):
        commands[name] = sub.add_parser(name, help=summary, description=COMMAND_HELP[name], epilog=epilog,
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
        if instance:
            commands[name].add_argument("name", help="instance name or ID")
        return commands[name]

    catalog = load_packages()
    oses = supported_os(catalog)
    titles = {name: package["title"] + (" (always)" if package["required"] else "") for name, package in catalog.items()}
    width = max(37, *map(len, titles.values()))
    rows, group = [f"  {'':<{width + 18}}" + "".join(f"{pretty_os(os_name):>14}" for os_name in oses)], None
    for name, package in catalog.items():
        if package["group"] != group:
            group = package["group"]
            rows.append(f"  {group}")
        levels = [support(package, os_name) for os_name in oses]
        rows.append(f"    {name:<14} {titles[name]:<{width}}" + "".join(f"{'-' if level.startswith('no') else level:>14}"
                                                                     for level in levels))
    p = command("start", "start a GPU instance (asks for name, flavor, software, image and key)",
                epilog="software packages (--packages) and compatibility (- = not supported):\n" + "\n".join(rows)
                + "\n\nDependencies are added automatically; each package is described in cloud-init/packages/.")
    p.add_argument("name", nargs="?", help="default: <username>-gpu")
    p.add_argument("--flavor", help=f"default: {DEFAULT_FLAVOR}")
    p.add_argument("--image", help="image or snapshot, name or ID (default: the best public image for the software)")
    p.add_argument("--packages", metavar="LIST", help="software, comma-separated, e.g. cuda,gcc-offload (see below)")
    p.add_argument("--burn", type=int, metavar="SECONDS", help="run gpu-burn for SECONDS once the instance is ready")
    p.add_argument("--burn-args", default="", metavar="ARGS", help='extra gpu_burn options, e.g. --burn-args="-d -tc"')
    p.add_argument("--key", metavar="FILE.pub", help="SSH public key to use, or 'new' to generate one")
    p.add_argument("--network", help="default: <username>-net, the network of your project on this cloud")
    p.add_argument("--userdata", metavar="FILE", help="own cloud-init file instead of the software packages")
    p.add_argument("--yes", action="store_true", help="do not ask, take flags and defaults (automatic without a terminal)")
    p.add_argument("--no-wait", action="store_true", help="do not wait until the software is installed")
    p.add_argument("--ready-timeout", type=int, metavar="MINUTES",
                   help="default: 60, or 720 if Spack packages are installed (not if the image has them)")
    p.add_argument("--timeout", type=int, default=600, help="seconds to wait for ACTIVE (default: 600)")
    p.set_defaults(func=cmd_start, catalog=catalog)

    p = command("list", "list your instances and the project quota")
    p.add_argument("--all", action="store_true", help="also show the instances of other users in the project")
    p.set_defaults(func=cmd_list)

    p = command("ssh", "log in to an instance or run a command there", instance=True)
    p.add_argument("command", nargs=argparse.REMAINDER, help="command to run instead of a login shell")
    p.set_defaults(func=cmd_ssh)

    p = command("logs", "follow the installation log of an instance", instance=True)
    p.set_defaults(func=cmd_logs)

    p = command("snapshot", "save an instance as an image for the image menu", instance=True)
    p.add_argument("image", help="name of the new image")
    p.add_argument("--force", action="store_true", help="snapshot even if the instance is not ready")
    p.add_argument("--share", action="store_true",
                   help="first remove your SSH keys, histories and cloud-init state, for an image for others")
    p.set_defaults(func=cmd_snapshot)

    p = command("shelve", "shelve an instance: keep its disk, release its GPU", instance=True)
    p.set_defaults(func=cmd_shelve)

    p = command("unshelve", "resume a shelved instance", instance=True)
    p.set_defaults(func=cmd_unshelve)

    p = command("rename", "rename an instance and its hostname", instance=True)
    p.add_argument("new_name", metavar="NEW_NAME", help="new name")
    p.set_defaults(func=cmd_rename)

    p = command("delete", "delete an instance and release its floating IP", instance=True)
    p.add_argument("--yes", action="store_true", help="do not ask (automatic without a terminal)")
    p.set_defaults(func=cmd_delete)

    command("setup", "create clouds.yaml (username and project, no password)")
    command("logout", "delete the saved login token")

    p = sub.add_parser("help", help="show this help, or the help of a command")
    p.add_argument("topic", nargs="?", choices=list(commands), metavar="COMMAND", help=", ".join(commands))

    args = parser.parse_args()
    if args.action is None or args.action == "help":
        (commands[args.topic] if getattr(args, "topic", None) else parser).print_help()
        return
    if args.action == "start":
        if args.burn is not None and args.burn <= 0:
            parser.error("--burn needs a positive number of seconds")
        if args.burn_args and args.burn is None:
            parser.error("--burn-args needs --burn")
        if args.userdata and (args.packages or args.burn is not None):
            parser.error("--userdata replaces the software packages; do not combine it with --packages or --burn")
        try:
            shlex.split(args.burn_args)
        except ValueError as e:
            parser.error(f"--burn-args: {e}")
    if args.debug:
        openstack.enable_logging(debug=True)
    with friendly_errors():
        if args.action == "setup":
            setup_clouds_yaml()
        elif args.action == "logout":
            logout()
        else:
            args.func(connect(TOKEN_MINUTES.get(args.action, DEFAULT_TOKEN_MINUTES)), args)


if __name__ == "__main__":
    main()
