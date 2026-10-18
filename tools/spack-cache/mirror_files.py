#!/usr/bin/env python3
"""The files of a Spack binary mirror (layout v3) that its index uses: the index manifest and the index,
layout.json, the spec manifests of the indexed specs and the blobs they reference. Everything else is
left over from older builds. Standard library only, because update.sh runs it on the build host; prune.py
uses referenced() for the Swift container.

Usage: mirror_files.py MIRROR_DIR [HAVE]
Prints the used files that exist, relative to MIRROR_DIR. With HAVE (a list of the files another copy of
the mirror already has, e.g. the Swift container), it prints only what that copy lacks, and leaves out the
blobs of spec manifests it has: a package that the build installed from the cache and pushed again gets a
new tarball with another checksum, while the other copy's manifest still names its old tarball.
"""
import json
import os
import sys
from pathlib import Path

INDEX = "v3/manifests/index/index.manifest.json"
SUFFIX = ".spec.manifest.json"


def blob(checksum):
    return f"blobs/sha256/{checksum[:2]}/{checksum}"


def referenced(read, names, have=frozenset()):
    """The files the index uses; read(name) returns a file's parsed JSON, names are all files of the mirror.
    Spec manifests in `have` count as present with their blobs, so neither is returned."""
    index_blob = blob(read(INDEX)["data"][0]["checksum"])
    hashes = set(read(index_blob)["database"]["installs"])
    used = {INDEX, index_blob, "v3/layout.json"}
    for name in names:
        if name.startswith("v3/manifests/spec/") and name.endswith(SUFFIX) and name not in have:
            if name[:-len(SUFFIX)].rsplit("-", 1)[-1] in hashes:  # .../<name>-<version>-<hash>.spec.manifest.json
                used.add(name)
                used.update(blob(entry["checksum"]) for entry in read(name)["data"])
    return used


if __name__ == "__main__":
    have = set(Path(sys.argv[2]).read_text(encoding="utf-8").split()) if len(sys.argv) > 2 else set()
    os.chdir(sys.argv[1])
    names = {os.path.relpath(os.path.join(root, f)) for root, _, files in os.walk(".") for f in files}
    with_json = lambda name: json.loads(Path(name).read_text(encoding="utf-8"))
    for name in sorted((referenced(with_json, names, have) & names) - have):
        print(name)
