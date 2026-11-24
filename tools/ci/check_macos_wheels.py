"""Fail unless each macOS wheel bundles a libomp that supports MACOSX_DEPLOYMENT_TARGET."""

import os
import struct
import sys
import zipfile

LC_BUILD_VERSION = 0x32


def macos_minimum(dylib: bytes) -> tuple[int, int]:
    ncmds = struct.unpack_from("<I", dylib, 16)[0]
    offset = 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", dylib, offset)
        if cmd == LC_BUILD_VERSION:
            version = struct.unpack_from("<I", dylib, offset + 12)[0]
            return version >> 16, (version >> 8) & 0xFF
        offset += size
    raise SystemExit("libomp.dylib has no LC_BUILD_VERSION")


target = tuple(int(part) for part in os.environ["MACOSX_DEPLOYMENT_TARGET"].split("."))
tag = f"macosx_{target[0]}_{target[1]}_"
for wheel in sys.argv[1:]:
    with zipfile.ZipFile(wheel) as z:
        libomp = [name for name in z.namelist() if name.endswith(".dylibs/libomp.dylib")]
        if not libomp:
            sys.exit(f"{wheel}: libomp.dylib is not bundled, so QBLAS was built without OpenMP")
        minimum = macos_minimum(z.read(libomp[0]))
    if minimum > target:
        sys.exit(f"{wheel}: bundled libomp needs macOS {minimum[0]}.{minimum[1]}")
    if tag not in os.path.basename(wheel):
        sys.exit(f"{wheel}: expected a {tag} wheel")
    print(f"{wheel}: libomp needs macOS {minimum[0]}.{minimum[1]}")
