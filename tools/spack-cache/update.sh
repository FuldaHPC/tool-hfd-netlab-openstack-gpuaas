#!/bin/sh
# Rebuilds the Spack binary cache on the build host and uploads it to Swift, in one command (see README.md).
# Run it locally from the repository, with your OpenStack login like gpuaas.py:
#   GPUAAS_BUILD_HOST=myhost tools/spack-cache/update.sh [--fresh] [--no-upload] [--prune]
#   --fresh      start with an empty mirror/ (after changing the spack-packages commit), so that the index
#                lists only the current specs; without it, packages already in mirror/ are reused
#   --no-upload  only build and test
#   --prune      after the upload, remove the files the new index no longer uses (prune.py)
# GPUAAS_BUILD_HOST is the ssh host that builds (Linux with Docker, no GPU needed); GPUAAS_BUILD_DIR is the
# folder there (default: gpuaas-cachebuild), which keeps mirror/ between runs.
set -eu
cd "$(dirname "$0")/../.."
fresh=0 upload=1 prune=0
for option in "$@"; do
  case $option in
    --fresh) fresh=1 ;;
    --no-upload) upload=0 ;;
    --prune) prune=1 ;;
    *) echo "usage: GPUAAS_BUILD_HOST=host $0 [--fresh] [--no-upload] [--prune]" >&2; exit 2 ;;
  esac
done
host=${GPUAAS_BUILD_HOST:-}
dir=${GPUAAS_BUILD_DIR:-gpuaas-cachebuild}
if [ -z "$host" ]; then
  echo "Set GPUAAS_BUILD_HOST to the ssh host that builds the cache (Linux with Docker; see README.md)." >&2
  exit 2
fi

echo "== build script and workspace to $host:$dir"
tools/spack-cache/generate.py | ssh "$host" "mkdir -p $dir && cat > $dir/build.sh"
ssh "$host" "cat > $dir/mirror_files.py" < tools/spack-cache/mirror_files.py
tar --no-xattrs -cf - workspace | ssh "$host" "rm -rf $dir/workspace && tar -C $dir -xf -"

echo "== build in an ubuntu:24.04 container (about 15 min from the cache, 2 h from scratch)"
# (The remote commands avoid zsh specials such as $status: the login shell of the build host may be zsh.)
ssh "$host" "cd $dir && docker rm -f gpuaas-spack-build >/dev/null 2>&1 || true
  if [ $fresh = 1 ]; then docker run --rm -v \"\$PWD\":/work ubuntu:24.04 rm -rf /work/mirror; fi
  docker run -d --name gpuaas-spack-build -v \"\$PWD\":/work ubuntu:24.04 bash -c 'bash /work/build.sh > /work/build.log 2>&1' >/dev/null
  rc=\$(docker wait gpuaas-spack-build); docker rm gpuaas-spack-build >/dev/null
  grep -E '^TIME|^WORKSPACE|^=== ' build.log
  # Pushing reports build-only packages (util-macros, gzip, ...) as not installed, which is harmless.
  grep -E '^==> Error' -A3 build.log | grep -vE 'PackageNotInstalledError|errors occurred while pushing|^--' | head -12
  exit \$rc"
# Chapel cannot link without the NVIDIA driver (-lcuda), so its workspace test fails in the container.
if ssh "$host" "grep '^WORKSPACE FAIL' $dir/build.log | grep -v 'WORKSPACE FAIL chapel'"; then
  echo "A workspace test failed; nothing is uploaded." >&2
  exit 1
fi
[ $upload = 1 ] || exit 0

echo "== upload the files of the new index that Swift does not have yet"
log=$(mktemp)
trap 'rm -f "$log"' EXIT
# A fresh token that stays valid for 55 minutes, so that it does not expire during a long upload, and the
# container URL; the token only goes to a file readable by you on the build host and is removed afterwards.
credentials=$(tools/spack-cache/prune.py --token 55)
printf '%s\n' "$credentials" |
ssh "$host" "umask 077; read -r T; read -r U; H=\$(mktemp); HAVE=\$(mktemp); printf 'X-Auth-Token: %s\n' \"\$T\" > \"\$H\"
  cd $dir/mirror
  curl -skL \"\$U/\" > \"\$HAVE\"
  python3 ../mirror_files.py . \"\$HAVE\" | while read -r f; do [ -n \"\$f\" ] || continue
    echo \"\$(curl -sk -o /dev/null -w '%{http_code}' -X PUT -H @\"\$H\" -T \"\$f\" \"\$U/\$f\") \$f\"
  done
  curl -sk -o /dev/null -w 'index: %{http_code}\n' -X PUT -H @\"\$H\" -T v3/manifests/index/index.manifest.json \"\$U/v3/manifests/index/index.manifest.json\"
  rm -f \"\$H\" \"\$HAVE\"" | tee "$log"
# Every line must report 201 (created), and the index must have been uploaded last.
if ! grep -q '^index: 201$' "$log" || grep -qvE '^(201 |index: 201$)' "$log"; then
  echo "Some uploads failed (the lines without 201 above); run the script again." >&2
  exit 1
fi

if [ $prune = 1 ]; then
  echo "== remove files the new index no longer uses"
  tools/spack-cache/prune.py --delete
fi
