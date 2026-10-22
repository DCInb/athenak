"""Memory-mapped, full-volume AthenaK 1.1 input for radiation diagnostics.

The record layout follows vis/python/bin_convert.py. Mapping the records avoids
loading velocities, ion temperatures, and radiation moments that are not needed
for the formal transfer calculation. Only native, unsliced active cells are used.
"""

from pathlib import Path

import numpy as np


def input_header(lines):
    blocks, block = {}, None
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line.startswith("<") and line.endswith(">"):
            block = line[1:-1]
            blocks[block] = {}
        elif line and "=" in line and block is not None:
            key, value = line.split("=", 1)
            blocks[block][key.strip()] = value.strip()
    return blocks


def unit_scales(header):
    units = header["units"]
    length, mass, time, mu = (float(units[key]) for key in
                              ("length_cgs", "mass_cgs", "time_cgs", "mu"))
    if not np.all(np.isfinite([length, mass, time, mu])) or min(length, mass, time, mu) <= 0:
        raise ValueError("Unit scales and mu must be finite and positive")
    # Match src/units, including its atomic mass and Boltzmann constants.
    temperature = (length / time)**2 * mu * 1.660538921e-24 / 1.3806488e-16
    return {"length_cm": length, "length_um": length * 1e4,
            "density_gcc": mass / length**3, "time_ns": time * 1e9,
            "temperature_kelvin": temperature}


def read_volume(path, variables):
    """Return metadata and read-only (block,z,y,x) views of requested fields."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.readline().strip() != b"Athena binary output version=1.1":
            raise ValueError(f"{path}: expected AthenaK binary format 1.1")
        count = int(stream.readline().split(b"=", 1)[1])
        parameters = {}
        for _ in range(count - 1):
            key, value = stream.readline().decode().split("=", 1)
            parameters[key.strip()] = value.strip()
        if "coarsening factor" in parameters:
            raise ValueError("Radiation tracing requires native, uncoarsened volumes")
        nvars = int(stream.readline().split(b"=", 1)[1])
        names = stream.readline().decode().split()[1:]
        size = int(stream.readline().split(b"=", 1)[1])
        if size <= 0 or size > path.stat().st_size - stream.tell():
            raise ValueError(f"{path}: invalid input-header size")
        header = input_header(stream.read(size).decode().splitlines())
        offset = stream.tell()
    if len(names) != nvars or len(set(names)) != nvars:
        raise ValueError(f"{path}: invalid variable list")
    missing = set(variables) - set(names)
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}; use fluid_w/fluid_3t outputs")
    loc_size, var_size = (int(parameters[f"size of {kind}"])
                          for kind in ("location", "variable"))
    if loc_size not in (4, 8) or var_size not in (4, 8):
        raise ValueError(f"{path}: unsupported numeric precision")
    mesh, block = header["mesh"], header["meshblock"]
    shape = np.array([int(block[f"nx{i}"]) for i in (1, 2, 3)])
    if np.any(shape < 1):
        raise ValueError("Invalid MeshBlock dimensions")
    dtype = np.dtype([("index", "=i4", (6,)), ("logical", "=i4", (4,)),
                      ("geometry", f"=f{loc_size}", (6,)),
                      ("data", f"=f{var_size}", (nvars, *shape[::-1]))])
    payload = path.stat().st_size - offset
    if payload <= 0 or payload % dtype.itemsize:
        raise ValueError(f"{path}: incomplete or sliced volume (full 3D dumps are required)")
    records = np.memmap(path, mode="r", offset=offset, dtype=dtype,
                        shape=(payload // dtype.itemsize,))
    indices = records["index"].astype(np.int64) - int(mesh["nghost"])
    expected = np.column_stack((np.zeros(3, dtype=int), shape - 1)).ravel()
    if not np.all(indices == expected):
        raise ValueError(f"{path}: slices, ghost zones, and coarsened dumps are unsupported")
    raw = {"path": path.resolve(), "header": header,
           "time": float(parameters["time"]), "cycle": int(parameters["cycle"]),
           "n_mbs": len(records), "mb_index": indices,
           "mb_logical": np.array(records["logical"]),
           "mb_geometry": np.array(records["geometry"], dtype=float),
           "mb_data": {key: records["data"][:, names.index(key)] for key in variables}}
    if not np.isfinite(raw["time"]):
        raise ValueError(f"{path}: non-finite snapshot time")
    for i in (1, 2, 3):
        raw[f"Nx{i}"] = int(mesh[f"nx{i}"])
        raw[f"nx{i}_mb"] = int(block[f"nx{i}"])
        for side in ("min", "max"):
            raw[f"x{i}{side}"] = float(mesh[f"x{i}{side}"])
    return raw


def read_pair(fluid_path, thermal_path, extra_fluid_fields=()):
    # Read the fluid header first to find the material scalar range.
    fluid = read_volume(fluid_path, ("dens",))
    materials = fluid["header"]["materials"]
    count = int(materials["nmaterials"])
    start = int(materials.get("scalar_index", 0))
    if count < 1 or start < 0:
        raise ValueError("Invalid material scalar configuration")
    scalars = [f"s_{i:02d}" for i in range(start, start + count)]
    fields = list(dict.fromkeys(("dens", *scalars, *extra_fluid_fields)))
    fluid = read_volume(fluid_path, fields)
    thermal = read_volume(thermal_path, ("tele",))
    for key in ("time", "cycle", "n_mbs", "mb_index", "mb_logical", "mb_geometry",
                *(f"{prefix}{i}{suffix}" for i in (1, 2, 3)
                  for prefix, suffix in (("Nx", ""), ("nx", "_mb"),
                                         ("x", "min"), ("x", "max")))):
        if not np.array_equal(fluid[key], thermal[key]):
            raise ValueError(f"Fluid and thermal dumps have mismatched {key}")
    for section in ("units", "materials", "thermal_radiation"):
        if fluid["header"][section] != thermal["header"][section]:
            raise ValueError(f"Fluid and thermal dumps have mismatched <{section}>")
    return fluid, thermal, unit_scales(fluid["header"])


def paired_paths(run_dir, dump="latest"):
    """Resolve full-volume pairs; never silently pair different output numbers."""
    directory = Path(run_dir) / "bin"
    def discover(kind):
        found = {}
        for path in directory.glob(f"*.{kind}.*.bin"):
            base, _, number, _ = path.name.rsplit(".", 3)
            key = (base, int(number))
            if key in found:
                raise ValueError(f"Duplicate {kind} dump {key}")
            found[key] = path
        return found
    fluid, thermal = discover("fluid"), discover("three_t")
    keys = sorted(fluid.keys() | thermal.keys())
    if not keys:
        raise ValueError(f"No full-volume fluid/three_t dumps in {directory}")
    if len({key[0] for key in keys}) != 1:
        raise ValueError("Run directory contains multiple basenames")
    if dump == "latest":
        keys = keys[-1:]
    elif dump != "all":
        keys = [key for key in keys if key[1] == int(dump)]
        if not keys:
            raise ValueError(f"No dump numbered {dump}")
    for key in keys:
        if key not in fluid or key not in thermal:
            raise ValueError(f"Unpaired full-volume dump {key}; wait for both atomic outputs")
    return [(fluid[key], thermal[key]) for key in keys]
