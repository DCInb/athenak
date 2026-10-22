#!/usr/bin/env python3
"""Compare double-precision hydro-only restart states, without loading ghost metadata.

Supports the current native AthenaK layout (9 Real RegionSize, 19 int RegionIndcs)
and this case's single-file output. It refuses mismatched meshes/times/layouts.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def read_restart(path):
    with path.open("rb") as handle:
        header = handle.read(40960)
    start = header.index(b"<par_end>") + 10
    blocks, level = struct.unpack_from("=ii", header, start)
    mesh = struct.unpack_from("=19i", header, start+8+72)
    block = struct.unpack_from("=19i", header, start+8+72+76)
    time, dt, cycle = struct.unpack_from("=ddi", header, start+8+72+152)
    size_offset = start+8+72+152+20+blocks*20
    bytes_per_block, = struct.unpack_from("=Q", header, size_offset)
    payload = size_offset+8
    assert path.stat().st_size == payload+blocks*bytes_per_block
    ng, nx1, nx2, nx3 = block[:4]
    shape = (nx3+2*ng, nx2+2*ng, nx1+2*ng)
    cells = int(np.prod(shape))
    assert bytes_per_block % (cells*8) == 0
    nvar = bytes_per_block//(cells*8)
    assert nvar == 31, "Expected the CH/He 20-group collision's 31 conserved fields"
    values = np.memmap(str(path), mode="r", dtype="=f8", offset=payload,
                       shape=(blocks, nvar)+shape)
    return dict(time=time, dt=dt, cycle=cycle, blocks=blocks, level=level,
                mesh=mesh[:10], block=block[:10]), values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    a_info, a = read_restart(args.baseline)
    b_info, b = read_restart(args.optimized)
    for key in ("cycle", "blocks", "level", "mesh", "block"):
        assert a_info[key] == b_info[key], key
    assert np.isclose(a_info["time"], b_info["time"], rtol=1.0e-12, atol=1.0e-15)
    assert np.isclose(a_info["dt"], b_info["dt"], rtol=1.0e-9, atol=1.0e-15)
    rows = []
    for field in range(a.shape[1]):
        x, y = a[:, field], b[:, field]
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)), field
        delta = np.abs(y-x)
        scale = max(float(np.max(np.abs(x))), 1.0e-300)
        linf = float(np.max(delta))/scale
        l2 = float(np.linalg.norm((y-x).ravel())) / max(
            float(np.linalg.norm(x.ravel())), 1.0e-300)
        rows.append(dict(field=field, max_absolute_difference=float(np.max(delta)),
                         relative_linf=linf, relative_l2=l2))
    result = dict(baseline=a_info, optimized=b_info, fields=rows,
                  compared_values_including_ghosts=int(a.size),
                  max_relative_linf=max(r["relative_linf"] for r in rows),
                  max_relative_l2=max(r["relative_l2"] for r in rows))
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("Restart comparison:", result["compared_values_including_ghosts"], "values")
    print("Maximum field-relative Linf:", result["max_relative_linf"])
    print("Maximum field-relative L2:", result["max_relative_l2"])
    assert result["max_relative_linf"] < 1.0e-6
    assert result["max_relative_l2"] < 1.0e-7


if __name__ == "__main__":
    main()
