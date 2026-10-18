#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["openstacksdk>=4", "pyyaml>=6", "questionary>=2"]
# ///
"""Remove files from the Swift container of the binary cache that the current index no longer uses:
the upload adds the files of every build and deletes nothing, so old builds pile up (an LLVM blob alone
is up to 900 MB). Kept are the index, layout.json, the spec manifests of the specs in the index and the
blobs they reference; only other spec manifests and blobs are removed, nothing else.

Usage (with your OpenStack login, like gpuaas.py):
  tools/spack-cache/prune.py              # dry run: list what would be removed
  tools/spack-cache/prune.py --delete     # remove it, then check that every kept file still exists
  tools/spack-cache/prune.py --token 55   # print a login token valid for 55 minutes and the container URL
"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # no __pycache__ in the repository
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gpuaas  # noqa: E402 (needs the path above)
import mirror_files  # noqa: E402 (next to this script)

CONTAINER = "gpuaas-spack-cache"


def container_url(conn):
    return f"{conn.object_store.get_endpoint().rstrip('/')}/{CONTAINER}"


def prune(store, delete):
    read = lambda name: json.loads(store.download_object(name, container=CONTAINER))
    objects = {o.name: o.content_length or 0 for o in store.objects(CONTAINER)}
    keep = mirror_files.referenced(read, objects)
    specs = [n for n in keep if n.startswith("v3/manifests/spec/")]
    missing = sorted(name for name in keep if name not in objects)
    if missing:
        sys.exit("The cache is incomplete, so nothing is removed; missing: " + ", ".join(missing))
    remove = sorted(name for name in objects if name not in keep
                    and (name.startswith("blobs/") or name.startswith("v3/manifests/spec/")))
    others = sorted(name for name in objects if name not in keep and name not in remove)

    size = lambda names: sum(objects[n] for n in names) / 1e9
    print(f"{len(objects)} files ({size(objects):.1f} GB); the index uses {len(specs)} spec manifests; "
          f"kept {len(keep)} files ({size(keep):.1f} GB)")
    for name in remove:
        print(f"  {'remove' if delete else 'would remove'} {objects[name] / 1e6:9.1f} MB  {name}")
    for name in others:
        print(f"  left alone (unknown) {name}")
    print(f"{len(remove)} files, {size(remove):.1f} GB {'removed' if delete else 'to remove (dry run; --delete removes them)'}")
    if not delete:
        return
    for name in remove:
        store.delete_object(name, container=CONTAINER)
    left = {o.name for o in store.objects(CONTAINER)}
    lost = sorted(keep - left)
    if lost:
        sys.exit("Kept files are missing after the removal: " + ", ".join(lost))
    print(f"Checked: all {len(keep)} kept files are still there.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--delete", action="store_true", help="remove the files, then check the kept ones")
    action.add_argument("--token", type=int, metavar="MINUTES",
                        help="print a login token valid for MINUTES and the container URL, one per line (update.sh)")
    args = parser.parse_args()
    with gpuaas.friendly_errors():
        if args.token:
            conn = gpuaas.connect(args.token)
            print(f"{conn.auth_token}\n{container_url(conn)}")
        else:
            prune(gpuaas.connect(30).object_store, args.delete)


if __name__ == "__main__":
    main()
