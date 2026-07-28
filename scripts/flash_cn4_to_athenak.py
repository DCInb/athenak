#!/usr/bin/env python3
"""Convert a two-temperature FLASH IONMIX4/CN4 EOS to AthenaK native v2.

The input temperature axis is in eV and the density axis is ion number
density in cm^-3.  IONMIX4 pressure and specific-energy blocks are stored in
J/cm^3 and J/g, respectively.  The output uses AthenaK's CGS table units:
g/cm^3, K, erg/cm^3, and erg/g.

The IONMIX4 ion and electron pressure/energy blocks are summed to construct a
single-temperature EOS.  Sound speed, Gamma_1, and Gamma_3 - 1 are derived
from finite differences of that total EOS surface.  ``--two-temperature-output``
preserves the separate ion/electron pressure and caloric surfaces, while
``--opacity-output`` writes the Rosseland transport and Planck
absorption/emission payloads in AthenaK's native multigroup-opacity format.

The manual/log choice describes only how the two axes are encoded.  EOS and
opacity payload fields must contain linear values; log-transformed payloads
are not supported.
"""

from __future__ import annotations

import argparse
from array import array
from bisect import bisect_right
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, TextIO, Tuple


ATOMIC_MASS_UNIT_G = 1.66053906660e-24
EV_TO_K = 11604.518121550082
JOULE_TO_ERG = 1.0e7
CN4_FIELD_WIDTH = 12
CXX_INT_MAX = 2_147_483_647
ATHENAK_TABLE_FIELD_COUNT = 9

OMITTED_E_EXPONENT = re.compile(
    r"^(?P<mantissa>[+-]?(?:\d\.\d{6}|\.\d{6}))"
    r"(?P<exponent>[+-]\d{3})$"
)

EOS_FIELD_NAMES = (
    "zbar",
    "dzdt",
    "pion",
    "pele",
    "dpidt",
    "dpedt",
    "eion",
    "eele",
    "cvion",
    "cvele",
    "deidn",
    "deedn",
)
EOS_FIELDS_TO_KEEP = frozenset(("zbar", "pion", "pele", "eion", "eele"))


class ConversionError(ValueError):
    """An input-table or conversion error suitable for a CLI diagnostic."""


class FixedTokens:
    """Compact, lazily decoded view of fixed-width CN4 numeric fields."""

    __slots__ = ("_payload", "_view", "_line_starts", "_line_numbers")

    def __init__(
        self,
        payload: bytearray,
        line_starts: array,
        line_numbers: array,
    ) -> None:
        self._payload = payload
        self._view = memoryview(payload)
        self._line_starts = line_starts
        self._line_numbers = line_numbers

    def __len__(self) -> int:
        return len(self._payload) // CN4_FIELD_WIDTH

    def raw(self, index: int) -> memoryview:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * CN4_FIELD_WIDTH
        return self._view[start:start + CN4_FIELD_WIDTH]

    def text(self, index: int) -> str:
        return self.raw(index).tobytes().decode("ascii")

    def location(self, index: int) -> str:
        line_index = bisect_right(self._line_starts, index) - 1
        token_in_line = index - self._line_starts[line_index]
        line = self._line_numbers[line_index]
        column = token_in_line * CN4_FIELD_WIDTH + 1
        return f"line {line}, column {column}"


@dataclass
class IonmixTable:
    ntemperature: int
    ndensity: int
    ngroups: int
    grid_mode: str
    has_electron_entropy: bool
    temperature_ev: List[float]
    number_density_cm3: List[float]
    fields: Dict[str, List[List[float]]]
    opacity_group_bounds_ev: List[float] = field(default_factory=list)
    opacities: Dict[str, List[List[List[float]]]] = field(default_factory=dict)


@dataclass
class AthenaKTable:
    density: List[float]
    temperature: List[float]
    pressure: List[List[float]]
    specific_energy: List[List[float]]
    sound_speed_squared: List[List[float]]
    gamma1: List[List[float]]
    gamma3m1: List[List[float]]
    zbar: List[List[float]]
    abar: List[List[float]]
    energy_offset_erg_g: float


@dataclass
class AthenaKOpacityTable:
    density: List[float]
    temperature: List[float]
    group_bounds: List[float]
    opacities: Dict[str, List[List[List[float]]]]


@dataclass
class AthenaKTwoTemperatureTable:
    density: List[float]
    temperature: List[float]
    abar: float
    ion_pressure: List[List[float]]
    electron_pressure: List[List[float]]
    ion_specific_energy: List[List[float]]
    electron_specific_energy: List[List[float]]
    mean_ionization: List[List[float]]


def _parse_header_integer(text: str, label: str) -> int:
    try:
        value = int(text.strip())
    except ValueError as exc:
        raise ConversionError(
            f"CN4 {label} field {text!r} is not an integer"
        ) from exc
    return value


def _read_cn4_tokens(path: Path) -> Tuple[int, int, FixedTokens]:
    try:
        source = path.open("r", encoding="ascii")
    except OSError as exc:
        raise ConversionError(f"cannot open input {path}: {exc}") from exc

    with source:
        dimensions_line = source.readline()
        if len(dimensions_line.rstrip("\r\n")) < 20:
            raise ConversionError(
                "CN4 first line must contain two 10-character dimensions"
            )
        ntemperature = _parse_header_integer(
            dimensions_line[0:10], "temperature dimension"
        )
        ndensity = _parse_header_integer(
            dimensions_line[10:20], "density dimension"
        )
        if ntemperature < 2 or ndensity < 2:
            raise ConversionError(
                "AthenaK requires at least two temperature and two density points; "
                f"CN4 declares {ntemperature} and {ndensity}"
            )
        if ntemperature > CXX_INT_MAX or ndensity > CXX_INT_MAX:
            raise ConversionError(
                "CN4 dimensions must each fit a positive C++ int; "
                f"found {ntemperature} and {ndensity}"
            )
        table_size = ntemperature * ndensity
        if table_size > CXX_INT_MAX // ATHENAK_TABLE_FIELD_COUNT:
            raise ConversionError(
                "AthenaK requires 9*NDENS*NTEMP <= C++ INT_MAX "
                f"({CXX_INT_MAX}); found 9*{ndensity}*{ntemperature}"
            )

        # The reference IONMIX reader consumes the remainder of the dimensions
        # line plus two descriptive lines.  readline() above already consumed
        # that remainder, so discard the two composition-description lines.
        for description_number in range(1, 3):
            if source.readline() == "":
                raise ConversionError(
                    "CN4 header ended before composition description line "
                    f"{description_number}"
                )

        payload = bytearray()
        line_starts = array("Q")
        line_numbers = array("Q")
        for line_number, raw_line in enumerate(source, start=4):
            line = raw_line.rstrip("\r\n").rstrip(" \t")
            if not line:
                continue
            if "\t" in line:
                raise ConversionError(
                    f"CN4 numeric payload contains a tab on line {line_number}; "
                    "numeric fields must be fixed-width"
                )
            if len(line) % CN4_FIELD_WIDTH != 0:
                raise ConversionError(
                    f"CN4 numeric line {line_number} has {len(line)} characters; "
                    f"expected a multiple of {CN4_FIELD_WIDTH}"
                )
            line_starts.append(len(payload) // CN4_FIELD_WIDTH)
            line_numbers.append(line_number)
            payload.extend(line.encode("ascii"))

    if not payload:
        raise ConversionError("CN4 file contains no numeric payload after its header")
    return ntemperature, ndensity, FixedTokens(
        payload, line_starts, line_numbers
    )


def _token_float(tokens: FixedTokens, index: int, label: str) -> float:
    raw = tokens.raw(index)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        # Some Fortran writers use D exponents.  E12.6 also narrowly permits
        # omission of E when a signed three-digit exponent consumes its slot.
        text = raw.tobytes().decode("ascii").strip()
        normalized = text.replace("D", "E").replace("d", "e")
        omitted_e = OMITTED_E_EXPONENT.fullmatch(normalized)
        if omitted_e is not None:
            normalized = (
                omitted_e.group("mantissa")
                + "E"
                + omitted_e.group("exponent")
            )
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ConversionError(
                f"{label} at {tokens.location(index)} is not numeric: "
                f"{tokens.text(index)!r}"
            ) from exc
    if not math.isfinite(value):
        raise ConversionError(
            f"{label} at {tokens.location(index)} must be finite, found "
            f"{tokens.text(index)!r}"
        )
    return value


def _token_integer(tokens: FixedTokens, index: int, label: str) -> int:
    try:
        value = int(tokens.raw(index))
    except ValueError as exc:
        raise ConversionError(
            f"{label} at {tokens.location(index)} is not an integer: "
            f"{tokens.text(index)!r}"
        ) from exc
    return value


def _take_floats(
    tokens: FixedTokens, start: int, count: int, label: str
) -> Tuple[List[float], int]:
    end = start + count
    if end > len(tokens):
        raise ConversionError(
            f"unexpected end of CN4 payload while reading {label}: "
            f"need {count} values, only {max(0, len(tokens) - start)} remain"
        )
    return (
        [_token_float(tokens, index, label) for index in range(start, end)],
        end,
    )


def _take_matrix(
    tokens: FixedTokens,
    start: int,
    ndensity: int,
    ntemperature: int,
    label: str,
) -> Tuple[List[List[float]], int]:
    count = ndensity * ntemperature
    end = start + count
    if end > len(tokens):
        raise ConversionError(
            f"unexpected end of CN4 payload while reading {label}: "
            f"need {count} values, only {max(0, len(tokens) - start)} remain"
        )
    matrix: List[List[float]] = []
    cursor = start
    for _ in range(ndensity):
        row_end = cursor + ntemperature
        matrix.append(
            [
                _token_float(tokens, index, label)
                for index in range(cursor, row_end)
            ]
        )
        cursor = row_end
    return matrix, end


def _take_cube(
    tokens: FixedTokens,
    start: int,
    ngroups: int,
    ndensity: int,
    ntemperature: int,
    label: str,
) -> Tuple[List[List[List[float]]], int]:
    """Read a group-density-temperature block with temperature varying fastest."""
    cube: List[List[List[float]]] = []
    cursor = start
    for group in range(ngroups):
        matrix, cursor = _take_matrix(
            tokens,
            cursor,
            ndensity,
            ntemperature,
            f"{label} group {group}",
        )
        cube.append(matrix)
    return cube, cursor


def _validate_float_block(
    tokens: FixedTokens, start: int, count: int, label: str
) -> int:
    end = start + count
    if end > len(tokens):
        raise ConversionError(
            f"unexpected end of CN4 payload while reading {label}: "
            f"need {count} values, only {max(0, len(tokens) - start)} remain"
        )
    for index in range(start, end):
        _token_float(tokens, index, label)
    return end


def _expected_payload_size(
    prefix_size: int,
    table_size: int,
    ngroups: int,
    has_electron_entropy: bool,
    opacity_bound_count: int,
) -> int:
    eos_size = len(EOS_FIELD_NAMES) * table_size
    entropy_size = table_size if has_electron_entropy else 0
    opacity_size = opacity_bound_count + 3 * table_size * ngroups
    return prefix_size + eos_size + entropy_size + opacity_size


def _select_entropy_layout(
    actual_size: int,
    prefix_size: int,
    table_size: int,
    ngroups: int,
    electron_entropy: str,
) -> Tuple[bool, int]:
    entropy_options = {
        "auto": (False, True),
        "absent": (False,),
        "present": (True,),
    }[electron_entropy]
    # IONMIX4 formally stores ngroups+1 bounds.  opacplot2 emits two dummy
    # bounds for ngroups=0, so accept that one exact compatibility layout too.
    bound_options = (1, 2) if ngroups == 0 else (ngroups + 1,)
    layouts = []
    for has_entropy in entropy_options:
        for bound_count in bound_options:
            expected_size = _expected_payload_size(
                prefix_size,
                table_size,
                ngroups,
                has_entropy,
                bound_count,
            )
            layouts.append((expected_size, has_entropy, bound_count))

    matches = [
        (has_entropy, bound_count)
        for expected_size, has_entropy, bound_count in layouts
        if actual_size == expected_size
    ]
    if len(matches) == 1:
        return matches[0]
    expected = ", ".join(str(layout[0]) for layout in layouts)
    raise ConversionError(
        f"payload has {actual_size} fixed-width fields; expected one of "
        f"{expected} for the selected entropy and opacity-bound layout"
    )


def _pow10_grid(start: float, step: float, count: int, label: str) -> List[float]:
    values: List[float] = []
    for index in range(count):
        exponent = start + index * step
        try:
            value = 10.0 ** exponent
        except OverflowError as exc:
            raise ConversionError(
                f"log-grid {label}[{index}] overflows 10**({exponent:.17g})"
            ) from exc
        if not math.isfinite(value) or value <= 0.0:
            raise ConversionError(
                f"log-grid {label}[{index}] is not finite and positive: "
                f"10**({exponent:.17g})"
            )
        values.append(value)
    return values


def _parse_grid_candidate(
    tokens: FixedTokens,
    ntemperature: int,
    ndensity: int,
    grid_mode: str,
    electron_entropy: str,
) -> IonmixTable:
    table_size = ntemperature * ndensity
    if grid_mode == "manual":
        if len(tokens) < 1:
            raise ConversionError("manual-grid payload is missing the group count")
        ngroups = _token_integer(tokens, 0, "opacity group count")
        prefix_size = 1 + ntemperature + ndensity
    elif grid_mode == "log":
        if len(tokens) < 5:
            raise ConversionError(
                "log-grid payload needs four grid parameters and a group count"
            )
        ngroups = _token_integer(tokens, 4, "opacity group count")
        prefix_size = 5
    else:
        raise AssertionError(f"unexpected grid mode {grid_mode!r}")

    if ngroups < 0:
        raise ConversionError(
            f"opacity group count must be non-negative, found {ngroups}"
        )
    has_electron_entropy, opacity_bound_count = _select_entropy_layout(
        len(tokens), prefix_size, table_size, ngroups, electron_entropy
    )

    if grid_mode == "manual":
        cursor = 1
        temperature_ev, cursor = _take_floats(
            tokens, cursor, ntemperature, "manual temperature"
        )
        number_density_cm3, cursor = _take_floats(
            tokens, cursor, ndensity, "manual ion number density"
        )
    else:
        density_step = _token_float(tokens, 0, "log-density step")
        density_start = _token_float(tokens, 1, "initial log-density")
        temperature_step = _token_float(tokens, 2, "log-temperature step")
        temperature_start = _token_float(tokens, 3, "initial log-temperature")
        number_density_cm3 = _pow10_grid(
            density_start, density_step, ndensity, "ion number density"
        )
        temperature_ev = _pow10_grid(
            temperature_start, temperature_step, ntemperature, "temperature"
        )
        cursor = 5

    fields: Dict[str, List[List[float]]] = {}
    for field_name in EOS_FIELD_NAMES:
        label = f"EOS field {field_name}"
        if field_name in EOS_FIELDS_TO_KEEP:
            fields[field_name], cursor = _take_matrix(
                tokens, cursor, ndensity, ntemperature, label
            )
        else:
            cursor = _validate_float_block(
                tokens, cursor, table_size, label
            )

    if has_electron_entropy:
        cursor = _validate_float_block(
            tokens, cursor, table_size, "electron entropy"
        )

    opacity_group_bounds_ev, cursor = _take_floats(
        tokens, cursor, opacity_bound_count, "opacity group boundary"
    )
    opacities: Dict[str, List[List[List[float]]]] = {}
    for name, label in (
        ("transport", "Rosseland opacity"),
        ("absorption", "Planck absorption opacity"),
        ("emission", "Planck emission opacity"),
    ):
        opacities[name], cursor = _take_cube(
            tokens,
            cursor,
            ngroups,
            ndensity,
            ntemperature,
            label,
        )
    if cursor != len(tokens):
        raise ConversionError(
            f"internal layout error: {len(tokens) - cursor} unconsumed payload fields"
        )

    return IonmixTable(
        ntemperature=ntemperature,
        ndensity=ndensity,
        ngroups=ngroups,
        grid_mode=grid_mode,
        has_electron_entropy=has_electron_entropy,
        temperature_ev=temperature_ev,
        number_density_cm3=number_density_cm3,
        fields=fields,
        opacity_group_bounds_ev=opacity_group_bounds_ev,
        opacities=opacities,
    )


def read_ionmix4(
    path: Path, grid_mode: str = "auto", electron_entropy: str = "auto"
) -> IonmixTable:
    """Read the two-temperature EOS portion of an IONMIX4/CN4 file."""
    try:
        ntemperature, ndensity, tokens = _read_cn4_tokens(path)
    except UnicodeError as exc:
        raise ConversionError(f"CN4 input {path} is not ASCII text: {exc}") from exc
    modes = ("manual", "log") if grid_mode == "auto" else (grid_mode,)
    candidates: List[IonmixTable] = []
    errors: List[str] = []
    for mode in modes:
        try:
            candidates.append(
                _parse_grid_candidate(
                    tokens,
                    ntemperature,
                    ndensity,
                    mode,
                    electron_entropy,
                )
            )
        except ConversionError as exc:
            errors.append(f"{mode}: {exc}")

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ConversionError(
            "CN4 grid layout is ambiguous; select --grid-mode manual or log"
        )
    detail = "; ".join(errors)
    raise ConversionError(
        "could not parse the two-temperature, 12-EOS-block IONMIX4 layout"
        + (f" ({detail})" if detail else "")
    )


def _axis_direction(values: Sequence[float], label: str) -> int:
    for index, value in enumerate(values):
        if not math.isfinite(value) or value <= 0.0:
            raise ConversionError(
                f"{label}[{index}] must be finite and positive, found {value!r}"
            )
    increasing = all(
        values[index + 1] > values[index] for index in range(len(values) - 1)
    )
    decreasing = all(
        values[index + 1] < values[index] for index in range(len(values) - 1)
    )
    if increasing:
        return 1
    if decreasing:
        return -1
    raise ConversionError(
        f"{label} axis must be strictly monotonic so it can be written in "
        "increasing order"
    )


def _validate_increasing_axis(
    values: Sequence[float], label: str, require_positive: bool
) -> None:
    for index, value in enumerate(values):
        if not math.isfinite(value) or (require_positive and value <= 0.0):
            requirement = "finite and positive" if require_positive else "finite"
            raise ConversionError(
                f"{label}[{index}] must be {requirement}, found {value!r}"
            )
    for index in range(len(values) - 1):
        if not values[index + 1] > values[index]:
            raise ConversionError(
                f"{label} axis must be strictly increasing; indices {index} and "
                f"{index + 1} contain {values[index]:.17g} and "
                f"{values[index + 1]:.17g}"
            )


def _with_increasing_axes(table: IonmixTable) -> IonmixTable:
    """Return a copy with increasing axes, leaving the parsed table unchanged."""
    temperature_ev = list(table.temperature_ev)
    number_density_cm3 = list(table.number_density_cm3)
    fields = {
        name: [list(row) for row in matrix]
        for name, matrix in table.fields.items()
    }
    opacities = {
        name: [[list(row) for row in matrix] for matrix in cube]
        for name, cube in table.opacities.items()
    }
    temperature_direction = _axis_direction(temperature_ev, "temperature")
    density_direction = _axis_direction(number_density_cm3, "ion number density")

    if temperature_direction < 0:
        temperature_ev.reverse()
        for matrix in fields.values():
            for row in matrix:
                row.reverse()
        for cube in opacities.values():
            for matrix in cube:
                for row in matrix:
                    row.reverse()
    if density_direction < 0:
        number_density_cm3.reverse()
        for matrix in fields.values():
            matrix.reverse()
        for cube in opacities.values():
            for matrix in cube:
                matrix.reverse()
    return IonmixTable(
        ntemperature=table.ntemperature,
        ndensity=table.ndensity,
        ngroups=table.ngroups,
        grid_mode=table.grid_mode,
        has_electron_entropy=table.has_electron_entropy,
        temperature_ev=temperature_ev,
        number_density_cm3=number_density_cm3,
        fields=fields,
        opacity_group_bounds_ev=list(table.opacity_group_bounds_ev),
        opacities=opacities,
    )


def _finite_positive(value: float, label: str, density: int, temperature: int) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ConversionError(
            f"{label}[density={density}, temperature={temperature}] must be finite "
            f"and positive, found {value!r}"
        )


def _validate_temperature_monotonic(
    matrix: Sequence[Sequence[float]], label: str
) -> None:
    for density_index, row in enumerate(matrix):
        for temperature_index in range(len(row) - 1):
            if not row[temperature_index + 1] > row[temperature_index]:
                raise ConversionError(
                    f"{label} must increase strictly with temperature at every "
                    "density; "
                    f"density row {density_index}, temperature indices "
                    f"{temperature_index} and {temperature_index + 1} contain "
                    f"{row[temperature_index]:.17g} and "
                    f"{row[temperature_index + 1]:.17g}"
                )


def _derivative_stencils(
    axis: Sequence[float],
) -> List[Tuple[Tuple[Tuple[int, int], ...], Tuple[float, ...]]]:
    """Return stable local derivative stencils for a nonuniform axis.

    Interior points use the derivative of the local quadratic.  Boundaries
    use the adjacent secant instead of a one-sided quadratic: the latter can
    overshoot and produce a negative heat capacity even when every tabulated
    energy row is strictly increasing.
    """
    count = len(axis)
    if count == 2:
        weight = 1.0 / (axis[1] - axis[0])
        stencil = (((0, 1),), (weight,))
        return [stencil, stencil]

    stencils: List[
        Tuple[Tuple[Tuple[int, int], ...], Tuple[float, ...]]
    ] = []
    for index in range(count):
        if index == 0:
            weight = 1.0 / (axis[1] - axis[0])
            stencils.append((((0, 1),), (weight,)))
            continue
        elif index == count - 1:
            weight = 1.0 / (axis[-1] - axis[-2])
            stencils.append((((count - 2, count - 1),), (weight,)))
            continue
        left_step = axis[index] - axis[index - 1]
        right_step = axis[index + 1] - axis[index]
        total_step = left_step + right_step
        # This is the nonuniform quadratic derivative expressed as a
        # positive weighted average of its adjacent secants.
        left_weight = right_step / (total_step * left_step)
        right_weight = left_step / (total_step * right_step)
        stencils.append(
            (
                ((index - 1, index), (index, index + 1)),
                (left_weight, right_weight),
            )
        )
    return stencils


def _differentiate_temperature(
    matrix: Sequence[Sequence[float]],
    stencils: Sequence[
        Tuple[Tuple[Tuple[int, int], ...], Tuple[float, ...]]
    ],
) -> List[List[float]]:
    result: List[List[float]] = []
    for row in matrix:
        derivative_row = []
        for segments, weights in stencils:
            derivative_row.append(
                sum(
                    weight * (row[right] - row[left])
                    for (left, right), weight in zip(segments, weights)
                )
            )
        result.append(derivative_row)
    return result


def _differentiate_density(
    matrix: Sequence[Sequence[float]],
    stencils: Sequence[
        Tuple[Tuple[Tuple[int, int], ...], Tuple[float, ...]]
    ],
) -> List[List[float]]:
    ndensity = len(matrix)
    ntemperature = len(matrix[0])
    result = [[0.0] * ntemperature for _ in range(ndensity)]
    for density_index, (segments, weights) in enumerate(stencils):
        for temperature_index in range(ntemperature):
            result[density_index][temperature_index] = sum(
                weight
                * (
                    matrix[right][temperature_index]
                    - matrix[left][temperature_index]
                )
                for (left, right), weight in zip(segments, weights)
            )
    return result


def convert_to_athenak(
    table: IonmixTable,
    mass_per_ion_g: float,
    energy_offset_erg_g: float = 0.0,
) -> AthenaKTable:
    """Convert parsed IONMIX data to an AthenaK native-v2 table in CGS."""
    if not math.isfinite(mass_per_ion_g) or mass_per_ion_g <= 0.0:
        raise ConversionError("mass per ion must be finite and positive")
    if not math.isfinite(energy_offset_erg_g):
        raise ConversionError("specific-energy offset must be finite")

    table = _with_increasing_axes(table)
    density = [value * mass_per_ion_g for value in table.number_density_cm3]
    temperature = [value * EV_TO_K for value in table.temperature_ev]
    _validate_increasing_axis(density, "converted mass density", True)
    _validate_increasing_axis(temperature, "converted temperature", True)

    ndensity = table.ndensity
    ntemperature = table.ntemperature
    pressure = [[0.0] * ntemperature for _ in range(ndensity)]
    specific_energy = [[0.0] * ntemperature for _ in range(ndensity)]
    zbar = [list(row) for row in table.fields["zbar"]]
    for density_index in range(ndensity):
        for temperature_index in range(ntemperature):
            pressure_value = JOULE_TO_ERG * (
                table.fields["pion"][density_index][temperature_index]
                + table.fields["pele"][density_index][temperature_index]
            )
            energy_value = JOULE_TO_ERG * (
                table.fields["eion"][density_index][temperature_index]
                + table.fields["eele"][density_index][temperature_index]
            ) + energy_offset_erg_g
            _finite_positive(
                pressure_value, "total pressure", density_index, temperature_index
            )
            if not math.isfinite(energy_value) or energy_value <= 0.0:
                raise ConversionError(
                    "total specific internal energy"
                    f"[density={density_index}, temperature={temperature_index}] "
                    f"must be finite and positive, found {energy_value!r}; use "
                    "--energy-offset-erg-g to apply an explicit reference shift"
                )
            zbar_value = zbar[density_index][temperature_index]
            if not math.isfinite(zbar_value) or zbar_value < 0.0:
                raise ConversionError(
                    f"zbar[density={density_index}, temperature={temperature_index}] "
                    f"must be finite and non-negative, found {zbar_value!r}"
                )
            pressure[density_index][temperature_index] = pressure_value
            specific_energy[density_index][temperature_index] = energy_value

    _validate_temperature_monotonic(pressure, "total pressure")
    _validate_temperature_monotonic(
        specific_energy, "total specific internal energy"
    )

    log_density = [math.log(value) for value in density]
    log_temperature = [math.log(value) for value in temperature]
    _validate_increasing_axis(log_density, "log mass density", False)
    _validate_increasing_axis(log_temperature, "log temperature", False)
    density_stencils = _derivative_stencils(log_density)
    temperature_stencils = _derivative_stencils(log_temperature)

    # AthenaK interpolates the positive core fields in log space.  Derive the
    # dimensionless log slopes first, then recover dP/dlog(x) and de/dlog(x)
    # locally.  This is stable across a large dynamic range and exactly
    # preserves power-law surfaces such as an ideal gas on a logarithmic grid.
    log_pressure = [[math.log(value) for value in row] for row in pressure]
    log_specific_energy = [
        [math.log(value) for value in row] for row in specific_energy
    ]
    log_pressure_log_density = _differentiate_density(
        log_pressure, density_stencils
    )
    log_pressure_log_temperature = _differentiate_temperature(
        log_pressure, temperature_stencils
    )
    log_energy_log_density = _differentiate_density(
        log_specific_energy, density_stencils
    )
    log_energy_log_temperature = _differentiate_temperature(
        log_specific_energy, temperature_stencils
    )

    sound_speed_squared = [[0.0] * ntemperature for _ in range(ndensity)]
    gamma1 = [[0.0] * ntemperature for _ in range(ndensity)]
    gamma3m1 = [[0.0] * ntemperature for _ in range(ndensity)]
    for density_index in range(ndensity):
        rho = density[density_index]
        for temperature_index in range(ntemperature):
            p_value = pressure[density_index][temperature_index]
            e_value = specific_energy[density_index][temperature_index]
            de_dlog_temperature = (
                e_value
                * log_energy_log_temperature[density_index][temperature_index]
            )
            _finite_positive(
                de_dlog_temperature,
                "d(specific energy)/d(log temperature)",
                density_index,
                temperature_index,
            )

            # From de = T ds + P/rho^2 d(rho):
            #   Gamma_3 - 1 = (P/rho - de/dlog(rho)) / de/dlog(T)
            # and c_s^2 = [dP/dlog(rho) + dP/dlog(T)*(Gamma_3-1)]/rho.
            gamma3_value = (
                p_value / rho
                - e_value
                * log_energy_log_density[density_index][temperature_index]
            ) / de_dlog_temperature
            cs2_value = (
                p_value
                * log_pressure_log_density[density_index][temperature_index]
                + p_value
                * log_pressure_log_temperature[density_index][temperature_index]
                * gamma3_value
            ) / rho
            gamma1_value = rho * cs2_value / p_value
            _finite_positive(
                gamma3_value,
                "gamma3m1",
                density_index,
                temperature_index,
            )
            _finite_positive(
                cs2_value,
                "sound speed squared",
                density_index,
                temperature_index,
            )
            _finite_positive(
                gamma1_value, "gamma1", density_index, temperature_index
            )
            gamma3m1[density_index][temperature_index] = gamma3_value
            sound_speed_squared[density_index][temperature_index] = cs2_value
            gamma1[density_index][temperature_index] = gamma1_value

    abar_value = mass_per_ion_g / ATOMIC_MASS_UNIT_G
    if not math.isfinite(abar_value) or abar_value <= 0.0:
        raise ConversionError("Abar overflows or is not positive after mass conversion")
    abar = [[abar_value] * ntemperature for _ in range(ndensity)]
    return AthenaKTable(
        density=density,
        temperature=temperature,
        pressure=pressure,
        specific_energy=specific_energy,
        sound_speed_squared=sound_speed_squared,
        gamma1=gamma1,
        gamma3m1=gamma3m1,
        zbar=zbar,
        abar=abar,
        energy_offset_erg_g=energy_offset_erg_g,
    )


def convert_opacity_to_athenak(
    table: IonmixTable,
    mass_per_ion_g: float,
) -> AthenaKOpacityTable:
    """Convert IONMIX opacity coordinates to CGS while preserving cm^2/g values."""
    if not math.isfinite(mass_per_ion_g) or mass_per_ion_g <= 0.0:
        raise ConversionError("mass per ion must be finite and positive")
    if table.ngroups <= 0:
        raise ConversionError("CN4 table has no opacity groups to export")

    table = _with_increasing_axes(table)
    if len(table.opacity_group_bounds_ev) != table.ngroups + 1:
        raise ConversionError(
            "CN4 opacity boundary count does not equal group count plus one"
        )
    required = ("transport", "absorption", "emission")
    if any(name not in table.opacities for name in required):
        raise ConversionError("CN4 opacity payload is incomplete")

    density = [value * mass_per_ion_g for value in table.number_density_cm3]
    temperature = [value * EV_TO_K for value in table.temperature_ev]
    group_bounds = [value * EV_TO_K for value in table.opacity_group_bounds_ev]
    _validate_increasing_axis(density, "converted opacity mass density", True)
    _validate_increasing_axis(temperature, "converted opacity temperature", True)
    _validate_increasing_axis(group_bounds, "converted opacity group boundary", False)
    if group_bounds[0] < 0.0:
        raise ConversionError("opacity group boundaries must be non-negative")

    opacities = {
        name: [[list(row) for row in matrix] for matrix in table.opacities[name]]
        for name in required
    }
    for name, cube in opacities.items():
        if len(cube) != table.ngroups:
            raise ConversionError(f"{name} opacity group dimension is inconsistent")
        for group, matrix in enumerate(cube):
            if len(matrix) != table.ndensity:
                raise ConversionError(
                    f"{name} opacity density dimension is inconsistent in group {group}"
                )
            for density_index, row in enumerate(matrix):
                if len(row) != table.ntemperature:
                    raise ConversionError(
                        f"{name} opacity temperature dimension is inconsistent in "
                        f"group {group}, density {density_index}"
                    )
                for temperature_index, value in enumerate(row):
                    valid = math.isfinite(value) and (
                        value > 0.0 if name == "transport" else value >= 0.0
                    )
                    if not valid:
                        requirement = "positive" if name == "transport" else "non-negative"
                        raise ConversionError(
                            f"{name} opacity[group={group}, density={density_index}, "
                            f"temperature={temperature_index}] must be finite and "
                            f"{requirement}, found {value!r}"
                        )

    return AthenaKOpacityTable(
        density=density,
        temperature=temperature,
        group_bounds=group_bounds,
        opacities=opacities,
    )


def convert_two_temperature_to_athenak(
    table: IonmixTable,
    mass_per_ion_g: float,
) -> AthenaKTwoTemperatureTable:
    """Preserve separate IONMIX ion/electron caloric and pressure surfaces."""
    if not math.isfinite(mass_per_ion_g) or mass_per_ion_g <= 0.0:
        raise ConversionError("mass per ion must be finite and positive")
    table = _with_increasing_axes(table)
    density = [value * mass_per_ion_g for value in table.number_density_cm3]
    temperature = [value * EV_TO_K for value in table.temperature_ev]
    _validate_increasing_axis(density, "converted 2T mass density", True)
    _validate_increasing_axis(temperature, "converted 2T temperature", True)

    converted_fields = {
        "ion_pressure": [
            [JOULE_TO_ERG * value for value in row]
            for row in table.fields["pion"]
        ],
        "electron_pressure": [
            [JOULE_TO_ERG * value for value in row]
            for row in table.fields["pele"]
        ],
        "ion_specific_energy": [
            [JOULE_TO_ERG * value for value in row]
            for row in table.fields["eion"]
        ],
        "electron_specific_energy": [
            [JOULE_TO_ERG * value for value in row]
            for row in table.fields["eele"]
        ],
        "mean_ionization": [list(row) for row in table.fields["zbar"]],
    }
    for name, matrix in converted_fields.items():
        for density_index, row in enumerate(matrix):
            for temperature_index, value in enumerate(row):
                positive = name != "mean_ionization"
                valid = math.isfinite(value) and (
                    value > 0.0 if positive else value >= 0.0
                )
                if not valid:
                    requirement = "positive" if positive else "non-negative"
                    raise ConversionError(
                        f"{name}[density={density_index}, "
                        f"temperature={temperature_index}] must be finite and "
                        f"{requirement}, found {value!r}"
                    )
            if name in ("ion_specific_energy", "electron_specific_energy"):
                for temperature_index in range(len(row) - 1):
                    if row[temperature_index + 1] < row[temperature_index]:
                        raise ConversionError(
                            f"{name} must be non-decreasing with temperature; "
                            f"density row {density_index}, temperature indices "
                            f"{temperature_index} and {temperature_index + 1} "
                            "decrease"
                        )

    abar = mass_per_ion_g / ATOMIC_MASS_UNIT_G
    return AthenaKTwoTemperatureTable(
        density=density,
        temperature=temperature,
        abar=abar,
        ion_pressure=converted_fields["ion_pressure"],
        electron_pressure=converted_fields["electron_pressure"],
        ion_specific_energy=converted_fields["ion_specific_energy"],
        electron_specific_energy=converted_fields["electron_specific_energy"],
        mean_ionization=converted_fields["mean_ionization"],
    )


def _format_float(value: float) -> str:
    return f"{value:.17e}"


def _write_values(output: TextIO, values: Iterable[float], per_line: int = 4) -> None:
    line: List[str] = []
    for value in values:
        line.append(_format_float(value))
        if len(line) == per_line:
            output.write(" ".join(line) + "\n")
            line.clear()
    if line:
        output.write(" ".join(line) + "\n")


def _write_matrix(output: TextIO, matrix: Sequence[Sequence[float]]) -> None:
    for density_index, row in enumerate(matrix):
        output.write(f"# density row {density_index}\n")
        _write_values(output, row)


def _write_native_stream(output: TextIO, table: AthenaKTable) -> None:
    ndensity = len(table.density)
    ntemperature = len(table.temperature)
    output.write("# Converted from a two-temperature FLASH IONMIX4/CN4 table.\n")
    output.write("# Units: density g/cm^3, temperature K, pressure erg/cm^3,\n")
    output.write("#        specific energy and sound speed squared erg/g.\n")
    output.write(
        "# Specific-energy reference offset: "
        f"{_format_float(table.energy_offset_erg_g)} erg/g.\n"
    )
    output.write("athenak_eos_table 2\n")
    output.write(f"dimensions {ndensity} {ntemperature}\n")
    output.write("density\n")
    _write_values(output, table.density)
    output.write("temperature\n")
    _write_values(output, table.temperature)

    core_fields = (
        ("pressure", table.pressure),
        ("specific_internal_energy", table.specific_energy),
        ("sound_speed_squared", table.sound_speed_squared),
    )
    for name, matrix in core_fields:
        output.write(f"\n{name}\n")
        _write_matrix(output, matrix)

    material_fields = (
        ("gamma1", table.gamma1),
        ("gamma3m1", table.gamma3m1),
        ("zbar", table.zbar),
        ("abar", table.abar),
    )
    output.write(f"\nmaterial_fields {len(material_fields)}\n")
    for name, matrix in material_fields:
        output.write(f"{name}\n")
        _write_matrix(output, matrix)
    output.write("end\n")


def write_native_v2(path: Path, table: AthenaKTable, force: bool = False) -> None:
    """Atomically write an AthenaK native-v2 table."""
    parent = path.parent
    if not parent.is_dir():
        raise ConversionError(f"output directory does not exist: {parent}")
    output_exists = os.path.lexists(path)
    if output_exists and not force:
        raise ConversionError(
            f"output already exists: {path} (use --force to replace it)"
        )

    if output_exists and path.exists():
        output_mode = path.stat().st_mode & 0o777
    else:
        current_umask = os.umask(0)
        os.umask(current_umask)
        output_mode = 0o666 & ~current_umask

    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
        )
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as output:
            _write_native_stream(output, table)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_name, output_mode)
        if force:
            os.replace(temporary_name, path)
        else:
            # Publishing through a same-directory hard link is atomic and
            # fails if any concurrent writer created the destination first.
            os.link(temporary_name, path)
            os.unlink(temporary_name)
        temporary_name = ""
    except FileExistsError as exc:
        raise ConversionError(
            f"output already exists: {path} (use --force to replace it)"
        ) from exc
    except OSError as exc:
        raise ConversionError(f"cannot write output {path}: {exc}") from exc
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _write_opacity_stream(output: TextIO, table: AthenaKOpacityTable) -> None:
    ndensity = len(table.density)
    ntemperature = len(table.temperature)
    ngroups = len(table.group_bounds) - 1
    output.write("# Converted from a FLASH IONMIX4/CN4 opacity payload.\n")
    output.write("# Units: density g/cm^3, temperature and group bounds K,\n")
    output.write("#        transport/absorption/emission mass opacity cm^2/g.\n")
    output.write("# Stored order: group, density, temperature (temperature fastest).\n")
    output.write("athenak_opacity_table 1\n")
    output.write(f"dimensions {ndensity} {ntemperature} {ngroups}\n")
    output.write("density\n")
    _write_values(output, table.density)
    output.write("temperature\n")
    _write_values(output, table.temperature)
    output.write("group_bound\n")
    _write_values(output, table.group_bounds)
    for name in ("transport", "absorption", "emission"):
        output.write(f"\n{name}\n")
        for group, matrix in enumerate(table.opacities[name]):
            output.write(f"# group {group}\n")
            _write_matrix(output, matrix)
    output.write("end\n")


def write_opacity_v1(
    path: Path,
    table: AthenaKOpacityTable,
    force: bool = False,
) -> None:
    """Atomically write an AthenaK native-v1 multigroup opacity table."""
    parent = path.parent
    if not parent.is_dir():
        raise ConversionError(f"output directory does not exist: {parent}")
    output_exists = os.path.lexists(path)
    if output_exists and not force:
        raise ConversionError(
            f"output already exists: {path} (use --force to replace it)"
        )
    if output_exists and path.exists():
        output_mode = path.stat().st_mode & 0o777
    else:
        current_umask = os.umask(0)
        os.umask(current_umask)
        output_mode = 0o666 & ~current_umask

    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
        )
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as output:
            _write_opacity_stream(output, table)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_name, output_mode)
        if force:
            os.replace(temporary_name, path)
        else:
            os.link(temporary_name, path)
            os.unlink(temporary_name)
        temporary_name = ""
    except FileExistsError as exc:
        raise ConversionError(
            f"output already exists: {path} (use --force to replace it)"
        ) from exc
    except OSError as exc:
        raise ConversionError(f"cannot write opacity output {path}: {exc}") from exc
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _write_two_temperature_stream(
    output: TextIO,
    table: AthenaKTwoTemperatureTable,
) -> None:
    output.write("# Separate ion/electron FLASH IONMIX4/CN4 EOS surfaces.\n")
    output.write("# Units: density g/cm^3, temperature K, pressure erg/cm^3,\n")
    output.write("#        specific internal energy erg/g.\n")
    output.write("athenak_two_temperature_eos 1\n")
    output.write(f"dimensions {len(table.density)} {len(table.temperature)}\n")
    output.write(f"abar {_format_float(table.abar)}\n")
    output.write("density\n")
    _write_values(output, table.density)
    output.write("temperature\n")
    _write_values(output, table.temperature)
    for name, matrix in (
        ("ion_pressure", table.ion_pressure),
        ("electron_pressure", table.electron_pressure),
        ("ion_specific_internal_energy", table.ion_specific_energy),
        ("electron_specific_internal_energy", table.electron_specific_energy),
        ("mean_ionization", table.mean_ionization),
    ):
        output.write(f"\n{name}\n")
        _write_matrix(output, matrix)
    output.write("end\n")


def write_two_temperature_v1(
    path: Path,
    table: AthenaKTwoTemperatureTable,
    force: bool = False,
) -> None:
    """Atomically write separate ion/electron IONMIX surfaces."""
    parent = path.parent
    if not parent.is_dir():
        raise ConversionError(f"output directory does not exist: {parent}")
    output_exists = os.path.lexists(path)
    if output_exists and not force:
        raise ConversionError(
            f"output already exists: {path} (use --force to replace it)"
        )
    if output_exists and path.exists():
        output_mode = path.stat().st_mode & 0o777
    else:
        current_umask = os.umask(0)
        os.umask(current_umask)
        output_mode = 0o666 & ~current_umask

    temporary_name = ""
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(parent)
        )
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as output:
            _write_two_temperature_stream(output, table)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_name, output_mode)
        if force:
            os.replace(temporary_name, path)
        else:
            os.link(temporary_name, path)
            os.unlink(temporary_name)
        temporary_name = ""
    except FileExistsError as exc:
        raise ConversionError(
            f"output already exists: {path} (use --force to replace it)"
        ) from exc
    except OSError as exc:
        raise ConversionError(f"cannot write 2T output {path}: {exc}") from exc
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _positive_finite_argument(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _finite_argument(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite")
    return value


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a two-temperature FLASH IONMIX4 (.cn4) EOS table to "
            "AthenaK native ASCII v2 in CGS units."
        ),
        epilog=(
            "Configure AthenaK EOS with table_unit_system=cgs. Native opacity "
            "output uses CGS axes and values and therefore requires the matching "
            "code-unit scales in <thermal_radiation>. The log grid mode "
            "controls only axis encoding; it does not mean payload fields are "
            "log-transformed. Only linear EOS and opacity payload values are "
            "supported."
        ),
    )
    parser.add_argument("input", type=Path, help="input FLASH IONMIX4/CN4 file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help=(
            "output AthenaK native-v2 EOS table (omit with --validate-only or "
            "when exporting only a separate 2T/opacity output)"
        ),
    )
    parser.add_argument(
        "--opacity-output",
        type=Path,
        help="also write the CN4 opacity payload as AthenaK native-v1",
    )
    parser.add_argument(
        "--two-temperature-output",
        type=Path,
        help="also preserve separate ion/electron IONMIX EOS surfaces",
    )
    mass_group = parser.add_mutually_exclusive_group(required=True)
    mass_group.add_argument(
        "--abar",
        type=_positive_finite_argument,
        metavar="AMU",
        help="mean mass per ion in atomic mass units",
    )
    mass_group.add_argument(
        "--mass-per-ion",
        "--mass-per-ion-g",
        dest="mass_per_ion",
        type=_positive_finite_argument,
        metavar="GRAMS",
        help="mass per ion in grams",
    )
    parser.add_argument(
        "--grid-mode",
        "--grid",
        choices=("auto", "manual", "log"),
        default="auto",
        help=(
            "CN4 density/temperature axis encoding; log means base-10 axis "
            "starts/steps, not log-transformed field values (default: auto)"
        ),
    )
    parser.add_argument(
        "--electron-entropy",
        choices=("auto", "present", "absent"),
        default="auto",
        help="whether an optional electron-entropy block is present (default: auto)",
    )
    parser.add_argument(
        "--energy-offset-erg-g",
        type=_finite_argument,
        default=0.0,
        metavar="OFFSET",
        help=(
            "constant added to specific internal energy in erg/g; use an explicit "
            "positive shift for CN4 tables with a non-positive energy reference"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace an existing output file",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="parse, convert, and validate without writing an output file",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the conversion summary"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        if (
            not arguments.validate_only
            and arguments.output is None
            and arguments.opacity_output is None
            and arguments.two_temperature_output is None
        ):
            raise ConversionError(
                "an EOS, --two-temperature-output, or --opacity-output is "
                "required unless --validate-only is used"
            )
        if arguments.validate_only and (
            arguments.output is not None
            or arguments.opacity_output is not None
            or arguments.two_temperature_output is not None
        ):
            raise ConversionError("omit all output paths when using --validate-only")
        outputs = [
            path for path in (
                arguments.output,
                arguments.two_temperature_output,
                arguments.opacity_output,
            )
            if path is not None
        ]
        if any(arguments.input.resolve() == path.resolve() for path in outputs):
            raise ConversionError("input and output paths must be different")
        if len({path.resolve() for path in outputs}) != len(outputs):
            raise ConversionError("all output paths must be different files")
        if arguments.abar is not None:
            mass_per_ion_g = arguments.abar * ATOMIC_MASS_UNIT_G
        else:
            mass_per_ion_g = arguments.mass_per_ion

        ionmix = read_ionmix4(
            arguments.input,
            grid_mode=arguments.grid_mode,
            electron_entropy=arguments.electron_entropy,
        )
        converted = None
        if arguments.output is not None or arguments.validate_only:
            converted = convert_to_athenak(
                ionmix,
                mass_per_ion_g,
                energy_offset_erg_g=arguments.energy_offset_erg_g,
            )
        opacity_converted = None
        if arguments.opacity_output is not None or (
            arguments.validate_only and ionmix.ngroups > 0
        ):
            opacity_converted = convert_opacity_to_athenak(
                ionmix, mass_per_ion_g
            )
        two_temperature_converted = None
        if arguments.two_temperature_output is not None:
            two_temperature_converted = convert_two_temperature_to_athenak(
                ionmix, mass_per_ion_g
            )
        if arguments.output is not None:
            assert converted is not None
            write_native_v2(arguments.output, converted, force=arguments.force)
        if arguments.opacity_output is not None:
            assert opacity_converted is not None
            write_opacity_v1(
                arguments.opacity_output,
                opacity_converted,
                force=arguments.force,
            )
        if arguments.two_temperature_output is not None:
            assert two_temperature_converted is not None
            write_two_temperature_v1(
                arguments.two_temperature_output,
                two_temperature_converted,
                force=arguments.force,
            )
    except (ConversionError, OSError) as exc:
        print(f"{parser.prog}: error: {exc}", file=sys.stderr)
        return 2

    if not arguments.quiet:
        entropy = "with" if ionmix.has_electron_entropy else "without"
        abar_value = mass_per_ion_g / ATOMIC_MASS_UNIT_G
        if arguments.validate_only:
            action = f"validated {arguments.input}"
        else:
            action = "wrote " + " and ".join(str(path) for path in outputs)
        print(
            f"{action}: {ionmix.ndensity} densities x "
            f"{ionmix.ntemperature} temperatures, {ionmix.grid_mode} grid, "
            f"{entropy} electron entropy, Abar={abar_value:.17g}, "
            f"energy offset={arguments.energy_offset_erg_g:.17g} erg/g"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
