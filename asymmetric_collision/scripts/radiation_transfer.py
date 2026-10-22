"""Multigroup emission/absorption and the cell-wise formal transfer solution."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


KELVIN_PER_EV = 11604.518121550082
PLANCK_INTEGRAL = np.pi**4 / 15
# Physical Planck intensity: 2 k_B^4 / (h^3 c^2), in CGS, per steradian.
PLANCK_PREFACTOR = 2 * 1.380649e-16**4 / (6.62607015e-27**3 * 2.99792458e10**2)


def _planck_integrals(x):
    """Both integral(0,x) and integral(x,infinity) of t^3/(exp(t)-1)."""
    x = np.asarray(x, dtype=float)
    small = np.minimum(x, 0.5)
    x2 = small * small
    lower = small**3 * (1/3 - small/8 + x2/60 - x2**2/5040
                         + x2**3/272160 - x2**4/13305600)
    # Separate the Wien tail to avoid subtracting two nearly equal CDFs when
    # both photon bounds greatly exceed the electron temperature.
    flat = x.ravel()
    active = np.flatnonzero((flat >= 0.5) & (flat < 745))
    z = flat[active]
    z2, z3 = z*z, z*z*z
    q = np.exp(-z)
    qn = q.copy()
    tail = np.zeros(flat.size)
    for n in range(1, 65):
        term = qn * (z3/n + 3*z2/n**2 + 6*z/n**3 + 6/n**4)
        tail[active] += term
        keep = term > 1e-17 * tail[active]
        if not np.any(keep):
            break
        active, z, z2, z3, q = (values[keep] for values in (active, z, z2, z3, q))
        qn = qn[keep] * q
    tail = tail.reshape(x.shape)
    return (np.where(x < 0.5, lower, PLANCK_INTEGRAL - tail),
            np.where(x < 0.5, PLANCK_INTEGRAL - lower, tail))


def planck_groups(temperature_kelvin, bounds_kelvin):
    """Frequency-integrated B_g [erg s^-1 cm^-2 sr^-1], with no band renormalization.

    Bounds are photon energy divided by k_B, the AthenaK group convention.
    The final array axis is the photon group; zero temperature emits zero.
    """
    temperature = np.asarray(temperature_kelvin, dtype=float)
    bounds = np.asarray(bounds_kelvin, dtype=float)
    if (np.any(~np.isfinite(temperature)) or np.any(temperature < 0)
            or bounds.ndim != 1 or len(bounds) < 2 or np.any(~np.isfinite(bounds))
            or bounds[0] < 0 or np.any(np.diff(bounds) <= 0)):
        raise ValueError("Invalid Planck temperature or photon group bounds")
    x = bounds / np.where(temperature[..., None] > 0, temperature[..., None], 1)
    lower, tail = _planck_integrals(x)
    integral = np.where(x[..., :-1] >= 0.5,
                        tail[..., :-1] - tail[..., 1:], np.diff(lower, axis=-1))
    return PLANCK_PREFACTOR * temperature[..., None]**4 * np.maximum(integral, 0)


def formal_step(intensity, absorption, emissivity, length):
    """Exact constant-cell solution, including zero absorption and tiny depth.

    alpha and j must use the same length unit as ds. The result has the units
    of intensity. Inputs broadcast; callers supply ds[...,None] for groups.
    """
    tau = absorption * length
    phi = np.ones_like(tau, dtype=float)
    np.divide(-np.expm1(-tau), tau, out=phi, where=tau != 0)
    return intensity * np.exp(-tau) + emissivity * length * phi


@dataclass
class OpacityTable:
    density: np.ndarray
    temperature: np.ndarray
    bounds: np.ndarray
    values: dict
    interpolation: str = "geometric"
    coordinates: str = "log"

    @classmethod
    def read(cls, path, interpolation="geometric", coordinates="log"):
        if interpolation not in ("linear", "log", "geometric", "hybrid"):
            raise ValueError(f"Unsupported opacity interpolation: {interpolation}")
        if coordinates not in ("linear", "log"):
            raise ValueError(f"Unsupported opacity coordinates: {coordinates}")
        tokens = iter(" ".join(line.split("#", 1)[0]
                               for line in Path(path).read_text().splitlines()).split())
        def expect(label):
            if next(tokens) != label:
                raise ValueError(f"{path}: expected {label}")
        def numbers(count):
            return np.array([float(next(tokens).replace("D", "e").replace("d", "e"))
                             for _ in range(count)])
        try:
            expect("athenak_opacity_table")
            expect("1")
            expect("dimensions")
            nd, nt, ng = (int(next(tokens)) for _ in range(3))
            if min(nd, nt, ng) < 1 or nd * nt * ng > 100_000_000:
                raise ValueError(f"{path}: invalid opacity dimensions")
            expect("density")
            density = numbers(nd)
            expect("temperature")
            temperature = numbers(nt)
            expect("group_bound")
            bounds = numbers(ng + 1)
            values = {}
            for kind in ("transport", "absorption", "emission"):
                expect(kind)
                values[kind] = numbers(ng * nd * nt).reshape(ng, nd, nt)
            expect("end")
        except StopIteration as exc:
            raise ValueError(f"{path}: truncated opacity table") from exc
        if next(tokens, None) is not None:
            raise ValueError(f"{path}: extra data after opacity table")
        for name, axis in (("density", density), ("temperature", temperature),
                           ("group", bounds)):
            if np.any(~np.isfinite(axis)) or np.any(axis < 0) or np.any(np.diff(axis) <= 0):
                raise ValueError(f"{path}: invalid {name} axis")
        if density[0] <= 0 or (coordinates == "log" and temperature[0] <= 0):
            raise ValueError(f"{path}: logarithmic coordinates must be positive")
        for kind, data in values.items():
            if (np.any(~np.isfinite(data)) or np.any(data < 0)
                    or ((kind == "transport" or interpolation == "log") and np.any(data == 0))):
                raise ValueError(f"{path}: invalid {kind} opacity")
        return cls(density, temperature, bounds, values, interpolation, coordinates)

    def evaluate(self, kind, density, temperature, groups):
        """Match OpacityTableDevice::Locate/Get, including clamping and zeros."""
        def locate(axis, values):
            values = np.clip(values, axis[0], axis[-1])
            if len(axis) == 1:
                return np.zeros(values.shape, dtype=int), np.zeros_like(values), 0
            if self.coordinates == "log":
                axis, values = np.log(axis), np.log(values)
            i = np.clip(np.searchsorted(axis, values, side="right") - 1, 0, len(axis) - 2)
            return i, (values - axis[i]) / (axis[i + 1] - axis[i]), 1
        di, fd, dstep = locate(self.density, np.asarray(density))
        ti, ft, tstep = locate(self.temperature, np.asarray(temperature))
        data = self.values[kind][groups]
        corners = np.stack([data[:, di, ti], data[:, di, ti + tstep],
                            data[:, di + dstep, ti], data[:, di + dstep, ti + tstep]])
        weights = np.stack([(1-fd)*(1-ft), (1-fd)*ft, fd*(1-ft), fd*ft])[:, None, :]
        linear = np.sum(corners * weights, axis=0)
        if self.interpolation == "linear":
            return linear.T
        positive = np.all(corners > 0, axis=0)
        logs = np.log(np.where(corners > 0, corners, 1))
        geometric = np.exp(np.sum(logs * weights, axis=0))
        return np.where(positive, geometric, linear).T


class PlasmaEmission:
    """The simulation's partial-density mixture, evaluated one leaf block at a time."""

    def __init__(self, fluid, thermal, scales, run_dir, groups=None, table_dir=None,
                 material_selection=None, emission_tracers=None):
        self.fluid, self.thermal, self.scales = fluid, thermal, scales
        header = fluid["header"]
        radiation, materials = header["thermal_radiation"], header["materials"]
        if radiation.get("opacity_model") != "mixed-table":
            raise ValueError("This diagnostic requires thermal_radiation/opacity_model=mixed-table")
        ng = int(radiation["n_groups"])
        self.bounds = np.array([float(radiation[f"group_bound_{g}"]) for g in range(ng + 1)])
        if (np.any(~np.isfinite(self.bounds)) or self.bounds[0] < 0
                or np.any(np.diff(self.bounds) <= 0)):
            raise ValueError("Invalid radiation group bounds")
        self.groups = np.arange(ng) if groups is None else np.asarray(groups, dtype=int)
        if (self.groups.ndim != 1 or not len(self.groups) or np.any(self.groups < 0)
                or np.any(self.groups >= ng) or len(set(self.groups)) != len(self.groups)):
            raise ValueError(f"Select distinct, zero-based groups in [0, {ng - 1}]")
        self.tables, self.settings, self.provenance = [], [], []
        count, start = int(materials["nmaterials"]), int(materials.get("scalar_index", 0))
        self.scalars = [f"s_{i:02d}" for i in range(start, start + count)]
        names = [materials.get(f"material{m}_name", str(m)) for m in range(count)]
        self.material_indices = list(range(count))
        if material_selection is not None:
            self.material_indices = []
            for selector in material_selection:
                token = str(selector)
                matches = [m for m, name in enumerate(names) if name.casefold() == token.casefold()]
                if not matches and token.isdecimal() and int(token) < count:
                    matches = [int(token)]
                if len(matches) != 1:
                    raise ValueError(f"Unknown or ambiguous material {token!r}; available: "
                                     + ", ".join(f"{m}={name}" for m, name in enumerate(names)))
                self.material_indices.append(matches[0])
            if (not self.material_indices
                    or len(set(self.material_indices)) != len(self.material_indices)):
                raise ValueError("Select at least one distinct material")
        self.material_names = [names[m] for m in self.material_indices]
        self.emission_tracers = [] if emission_tracers is None else list(emission_tracers)
        if emission_tracers is not None:
            if len(self.material_indices) != 1:
                raise ValueError("Emission tracers require exactly one selected material (--materials)")
            if (not self.emission_tracers
                    or len(set(self.emission_tracers)) != len(self.emission_tracers)):
                raise ValueError("Select at least one distinct emission tracer")
            for key in self.emission_tracers:
                if not key.startswith("s_") or not key[2:].isdecimal() or key in self.scalars:
                    raise ValueError(f"Emission tracer {key!r} must be an origin scalar, not a material fraction")
                if key not in fluid["mb_data"]:
                    raise ValueError(f"Missing emission tracer {key!r} in fluid data")
        self.emission_selection = {
            "tracers": self.emission_tracers,
            "material": self.material_names[0] if self.emission_tracers else None,
            "weight_rule": ("clip(sum(clip(tracer,0,1))/clip(material_scalar,0,1),0,1); "
                            "zero where the material scalar is zero"
                            if self.emission_tracers else "1 (all origins)"),
            "absorption": "Unchanged: all origins of the selected materials absorb",
        }
        self.clamp_counts = [dict(material_index=m, material=names[m],
                                  enabled=m in self.material_indices, sampled_cells=0,
                                  density_low=0, density_high=0,
                                  temperature_low=0, temperature_high=0) for m in range(count)]
        for m in self.material_indices:
            prefix = f"material{m}_opacity_"
            filename = Path(materials[prefix + "table_file"])
            path = (Path(table_dir) / filename.name if table_dir is not None
                    else Path(run_dir) / filename)
            table = OpacityTable.read(path, materials.get(prefix + "interpolation", "linear"),
                                      materials.get(prefix + "coordinate_interpolation", "linear"))
            cfg = {key: float(materials.get(prefix + key, default)) for key, default in
                   (("density_scale", 1), ("temperature_scale", 1), ("group_bound_scale", 1),
                    ("absorption_scale", materials.get(prefix + "value_scale", 1)),
                    ("emission_scale", materials.get(prefix + "value_scale", 1)))}
            if (any(not np.isfinite(value) or value < 0 for value in cfg.values())
                    or min(cfg[key] for key in ("density_scale", "temperature_scale",
                                               "group_bound_scale")) <= 0):
                raise ValueError("Invalid opacity-table scales")
            if (len(table.bounds) != len(self.bounds)
                    or not np.allclose(table.bounds * cfg["group_bound_scale"], self.bounds,
                                       rtol=1e-8, atol=0)):
                raise ValueError(f"{path}: photon bounds disagree with the snapshot")
            self.tables.append(table)
            self.settings.append(cfg)
            self.provenance.append({"material_index": m, "material": names[m],
                                    "path": str(path.resolve()),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "interpolation": table.interpolation,
                                    "coordinates": table.coordinates, "scales": cfg})

    def coefficients(self, block):
        """Return alpha [code_length^-1] and j [CGS intensity/code_length]."""
        rho = np.asarray(self.fluid["mb_data"]["dens"][block], dtype=float).ravel()
        tele = np.asarray(self.thermal["mb_data"]["tele"][block], dtype=float).ravel()
        fractions = np.array([self.fluid["mb_data"][key][block].ravel()
                              for key in self.scalars], dtype=float)
        if (np.any(~np.isfinite(rho)) or np.any(rho < 0) or np.any(~np.isfinite(tele))
                or np.any(tele < 0) or np.any(~np.isfinite(fractions))):
            raise ValueError(f"Invalid density, electron temperature, or composition in block {block}")
        fractions = np.clip(fractions, 0, 1)
        emission_weight = None
        if self.emission_tracers:
            origins = np.array([self.fluid["mb_data"][key][block].ravel()
                                for key in self.emission_tracers], dtype=float)
            if np.any(~np.isfinite(origins)):
                raise ValueError(f"Invalid emission origin fractions in block {block}")
            material_fraction = fractions[self.material_indices[0]]
            emission_weight = np.zeros_like(rho)
            # Both origin and material fractions refer to total cell mass. Take
            # their ratio before normalizing the material mixture. This partitions
            # the source at the original CH density/temperature, keeping all CH
            # absorption, including that of nonemitting corona-origin matter.
            np.divide(np.clip(origins, 0, 1).sum(axis=0), material_fraction,
                      out=emission_weight, where=material_fraction > 0)
            np.clip(emission_weight, 0, 1, out=emission_weight)
        total = fractions.sum(axis=0)
        fractions /= np.where(total > 0, total, 1)
        fractions[-1, total == 0] = 1  # MaterialComposition's last-material fallback.
        alpha = np.zeros((rho.size, len(self.groups)))
        emission = np.zeros_like(alpha)
        # Normalize the full simulated mixture first, then select contributors.
        # In a CH-only calculation rho_CH stays rho*Y_CH, including in mixed cells.
        for m, table, cfg in zip(self.material_indices, self.tables, self.settings):
            partial = rho * fractions[m]
            selected = partial > 0
            d = partial[selected] * cfg["density_scale"]
            t = tele[selected] * cfg["temperature_scale"]
            counts = self.clamp_counts[m]
            counts["sampled_cells"] += int(selected.sum())
            for key, mask in (("density_low", d < table.density[0]),
                              ("density_high", d > table.density[-1]),
                              ("temperature_low", t < table.temperature[0]),
                              ("temperature_high", t > table.temperature[-1])):
                counts[key] += int(mask.sum())
            if not np.any(selected):
                continue
            for kind, target in (("absorption", alpha), ("emission", emission)):
                target[selected] += (partial[selected, None] * cfg[kind + "_scale"]
                                     * table.evaluate(kind, d, t, self.groups))
        # Use the physical Planck function, independent of any reduced simulation c.
        temperature_scale = self.scales["temperature_kelvin"]
        temperatures, inverse = np.unique(tele, return_inverse=True)
        first, last = int(self.groups.min()), int(self.groups.max())
        # Narrow-band images do not need the unselected low-energy Planck tails.
        blackbody = planck_groups(temperatures * temperature_scale,
                                 self.bounds[first:last + 2] * temperature_scale)
        blackbody = blackbody[inverse][:, self.groups - first]
        emissivity = emission * blackbody
        if emission_weight is not None:
            emissivity *= emission_weight[:, None]
        if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(emissivity)):
            raise ValueError(f"Radiation coefficients overflow in block {block}")
        return alpha, emissivity
