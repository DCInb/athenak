#!/usr/bin/env python3
"""Analyze laser/radiation communication, Biermann generation, and MPI invariance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
sys.path.insert(0, str(REPO / "vis" / "python"))
from bin_convert import read_binary  # noqa: E402


FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
EXPECTED_CASES = (
    "coupled_gpu1",
    "coupled_gpu2",
    "no_radiation_gpu1",
    "no_biermann_gpu1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-plots", action="store_true", help="skip matplotlib figure generation"
    )
    return parser.parse_args()


def parse_laser_log(path: Path) -> dict[str, float]:
    records = []
    for line in path.read_text().splitlines():
        if not line.startswith("laser:"):
            continue
        records.append(
            {
                key: float(value)
                for key, value in re.findall(
                    rf"([a-z_]+)=({FLOAT_PATTERN})", line
                )
            }
        )
    if not records:
        raise RuntimeError(f"No laser diagnostics found in {path}")
    illuminated = [record for record in records if record["launched"] > 0.0]
    if not illuminated:
        raise RuntimeError(f"No illuminated laser stages found in {path}")
    return {
        "diagnostic_count": len(records),
        "illuminated_count": len(illuminated),
        "max_residual": max(record["residual"] for record in records),
        "max_active": max(record["active"] for record in records),
        "max_reflected": max(record["reflected"] for record in records),
        "max_transfers": max(record["transfers"] for record in records),
        "max_dispersion": max(record["dispersion"] for record in records),
        "mean_deposited_fraction": float(
            np.mean(
                [record["deposited"] / record["launched"] for record in illuminated]
            )
        ),
        "mean_escaped_fraction": float(
            np.mean(
                [record["escaped"] / record["launched"] for record in illuminated]
            )
        ),
    }


def read_history(path: Path) -> dict[str, np.ndarray]:
    header_lines = [line for line in path.read_text().splitlines() if line.startswith("#")]
    if not header_lines:
        raise RuntimeError(f"No history header in {path}")
    labels = re.findall(r"\[\d+\]=([^\s]+)", header_lines[-1])
    values = np.loadtxt(path)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.shape[1] != len(labels):
        raise RuntimeError(
            f"History columns do not match header in {path}: "
            f"{values.shape[1]} != {len(labels)}"
        )
    return {label: values[:, index] for index, label in enumerate(labels)}


def assemble_uniform(path: Path) -> dict[str, object]:
    raw = read_binary(str(path))
    nx = (int(raw["Nx1"]), int(raw["Nx2"]), int(raw["Nx3"]))
    mb_nx = (int(raw["nx1_out_mb"]), int(raw["nx2_out_mb"]),
             int(raw["nx3_out_mb"]))
    fields = {
        name: np.zeros((nx[2], nx[1], nx[0]), dtype=np.float64)
        for name in raw["var_names"]
    }
    for block, logical in enumerate(raw["mb_logical"]):
        if int(logical[3]) != 0:
            raise RuntimeError("This benchmark analyzer expects a uniform mesh")
        i0 = int(logical[0]) * mb_nx[0]
        j0 = int(logical[1]) * mb_nx[1]
        k0 = int(logical[2]) * mb_nx[2]
        for name in raw["var_names"]:
            values = np.asarray(raw["mb_data"][name][block], dtype=np.float64)
            nz, ny, nx_local = values.shape
            fields[name][k0:k0+nz, j0:j0+ny, i0:i0+nx_local] = values
    return {
        "time": float(raw["time"]),
        "nx": nx,
        "extent": (
            float(raw["x1min"]),
            float(raw["x1max"]),
            float(raw["x2min"]),
            float(raw["x2max"]),
        ),
        "fields": fields,
    }


def final_binary(run_dir: Path, basename: str, output_id: str) -> Path:
    paths = sorted((run_dir / "bin").glob(f"{basename}.{output_id}.*.bin"))
    if not paths:
        raise RuntimeError(
            f"No {output_id} binary for {basename} under {run_dir / 'bin'}"
        )
    return paths[-1]


def final_minus_initial(history: dict[str, np.ndarray], name: str) -> float:
    return float(history[name][-1] - history[name][0])


def topology_metrics(fluid: dict[str, object]) -> dict[str, float]:
    fields = fluid["fields"]
    bz = np.asarray(fields["bcc3"][0], dtype=np.float64)
    reflected = -np.flip(bz, axis=0)
    denominator = math.sqrt(float(np.sum(bz*bz) * np.sum(reflected*reflected)))
    antisymmetry = float(np.sum(bz*reflected) / denominator) if denominator > 0.0 else 0.0
    x1min, x1max, x2min, x2max = fluid["extent"]
    ny, nx = bz.shape
    x1 = x1min + (np.arange(nx) + 0.5)*(x1max-x1min)/nx
    x2 = x2min + (np.arange(ny) + 0.5)*(x2max-x2min)/ny
    peak = np.unravel_index(np.argmax(np.abs(bz)), bz.shape)
    return {
        "bz_peak": float(np.max(np.abs(bz))),
        "bz_antisymmetry": antisymmetry,
        "bz_peak_x1": float(x1[peak[1]]),
        "bz_peak_x2": float(x2[peak[0]]),
    }


def compare_histories(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, object]:
    same_labels = reference.keys() == candidate.keys()
    max_difference = 0.0
    same = same_labels
    if same_labels:
        for name in reference:
            if reference[name].shape != candidate[name].shape:
                same = False
                continue
            difference = float(np.max(np.abs(reference[name]-candidate[name])))
            max_difference = max(max_difference, difference)
            same = same and np.allclose(
                reference[name], candidate[name], rtol=2.0e-9, atol=2.0e-9
            )
    return {"pass": bool(same), "max_abs_difference": max_difference}


def compare_datasets(
    reference: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    same = reference["nx"] == candidate["nx"]
    max_difference = 0.0
    reference_fields = reference["fields"]
    candidate_fields = candidate["fields"]
    same = same and reference_fields.keys() == candidate_fields.keys()
    if same:
        for name in reference_fields:
            difference = float(
                np.max(np.abs(reference_fields[name]-candidate_fields[name]))
            )
            max_difference = max(max_difference, difference)
            same = same and np.allclose(
                reference_fields[name],
                candidate_fields[name],
                rtol=2.0e-9,
                atol=2.0e-9,
            )
    return {"pass": bool(same), "max_abs_difference": max_difference}


def load_cases() -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    status_rows = list(csv.DictReader((BENCH_DIR / "run_status.csv").open()))
    if {row["case"] for row in status_rows} != set(EXPECTED_CASES):
        raise RuntimeError("run_status.csv does not contain the expected four cases")

    results = []
    loaded: dict[str, dict[str, object]] = {}
    for row in status_rows:
        case = row["case"]
        run_dir = Path(row["run_dir"])
        basename = f"laser_target_{case}"
        history_path = run_dir / f"{basename}.user.hst"
        history = read_history(history_path)
        laser_log = parse_laser_log(Path(row["log"]))
        datasets = {
            output_id: assemble_uniform(final_binary(run_dir, basename, output_id))
            for output_id in ("fluid", "three_t", "laser")
        }
        topology = topology_metrics(datasets["fluid"])
        laser_energy = float(history["laser_E"][-1])
        chain_gain = final_minus_initial(history, "chain_E")
        radiation_gain = final_minus_initial(history, "erad_E")
        electron_gain = final_minus_initial(history, "eele_E")
        ion_gain = final_minus_initial(history, "eion_E")
        source_integral = float(
            np.trapezoid(history["bier_S"], history["time"])
        )
        energy_error = (
            abs(chain_gain-laser_energy)/laser_energy
            if laser_energy > 0.0 else math.inf
        )
        result = {
            "case": case,
            "gpus": int(row["gpus"]),
            "run_ok": int(row["exit_code"]) == 0,
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "laser_energy": laser_energy,
            "chain_energy_gain": chain_gain,
            "chain_laser_relative_error": energy_error,
            "radiation_energy_gain": radiation_gain,
            "electron_energy_gain": electron_gain,
            "ion_energy_gain": ion_gain,
            "final_abs_bz_integral": float(history["abs_Bz"][-1]),
            "final_magnetic_energy": float(history["mag_E"][-1]),
            "biermann_source_time_integral": source_integral,
            "bz_to_source_integral": (
                float(history["abs_Bz"][-1])/source_integral
                if source_integral > 0.0 else 0.0
            ),
            "laser_deposition_centroid_x1": (
                float(history["laser_x"][-1]/laser_energy)
                if laser_energy > 0.0 else math.nan
            ),
            "radiation_gain_centroid_x1": (
                final_minus_initial(history, "erad_x")/radiation_gain
                if abs(radiation_gain) > 0.0 else math.nan
            ),
            "electron_gain_centroid_x1": (
                final_minus_initial(history, "eele_x")/electron_gain
                if abs(electron_gain) > 0.0 else math.nan
            ),
            **laser_log,
            **topology,
        }
        result["case_ok"] = bool(
            result["run_ok"]
            and result["max_residual"] < 1.0e-10
            and result["chain_laser_relative_error"] < 5.0e-2
        )
        results.append(result)
        loaded[case] = {
            "row": row,
            "history": history,
            "datasets": datasets,
            "result": result,
        }
    return results, loaded


def make_plots(loaded: dict[str, dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    plot_dir = BENCH_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    coupled = loaded["coupled_gpu1"]
    fluid = coupled["datasets"]["fluid"]
    three_t = coupled["datasets"]["three_t"]
    laser = coupled["datasets"]["laser"]
    extent = fluid["extent"]
    image_extent = (extent[0], extent[1], extent[2], extent[3])
    density = fluid["fields"]["dens"][0]
    bz = fluid["fields"]["bcc3"][0]
    tele = three_t["fields"]["tele"][0]
    trad = three_t["fields"]["trad"][0]
    path = laser["fields"]["laser_path"][0]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    density_image = axes[0, 0].imshow(
        density,
        origin="lower",
        extent=image_extent,
        aspect="auto",
        norm=LogNorm(vmin=max(float(np.min(density)), 1.0e-6),
                     vmax=float(np.max(density))),
        cmap="viridis",
    )
    if np.max(path) > 0.0:
        axes[0, 0].contour(
            np.linspace(extent[0], extent[1], path.shape[1]),
            np.linspace(extent[2], extent[3], path.shape[0]),
            path,
            levels=[0.05*np.max(path), 0.25*np.max(path)],
            colors=("white", "cyan"),
            linewidths=(0.7, 1.0),
        )
    fig.colorbar(density_image, ax=axes[0, 0], label="density")
    axes[0, 0].set_title("Solid/corona density and ray path")

    tele_image = axes[0, 1].imshow(
        tele, origin="lower", extent=image_extent, aspect="auto", cmap="inferno"
    )
    fig.colorbar(tele_image, ax=axes[0, 1], label=r"$T_e$")
    axes[0, 1].set_title("Laser-heated electron temperature")

    trad_image = axes[1, 0].imshow(
        trad, origin="lower", extent=image_extent, aspect="auto", cmap="magma"
    )
    fig.colorbar(trad_image, ax=axes[1, 0], label=r"$T_{rad}$")
    axes[1, 0].set_title("Thermal-radiation response")

    bmax = float(np.max(np.abs(bz)))
    bz_image = axes[1, 1].imshow(
        bz,
        origin="lower",
        extent=image_extent,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bmax, vcenter=0.0, vmax=bmax),
    )
    fig.colorbar(bz_image, ax=axes[1, 1], label=r"$B_3$")
    axes[1, 1].set_title("Biermann field (2D cut of toroidal topology)")
    for ax in axes.flat:
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    fig.savefig(plot_dir / "laser_target_fields.png", dpi=180)
    fig.savefig(plot_dir / "laser_target_fields.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
    for case, color in (("coupled_gpu1", "#2563eb"),
                        ("no_radiation_gpu1", "#d97706")):
        history = loaded[case]["history"]
        axes[0].plot(
            history["time"], history["erad_E"]-history["erad_E"][0],
            color=color, label=case,
        )
    axes[0].set_title("Laser → radiation communication")
    axes[0].set_ylabel(r"$\Delta E_{rad}$")
    axes[0].legend(fontsize=8)

    for case, color in (("coupled_gpu1", "#2563eb"),
                        ("no_biermann_gpu1", "#dc2626")):
        history = loaded[case]["history"]
        axes[1].plot(history["time"], history["abs_Bz"], color=color, label=case)
    axes[1].set_title("Crossed gradients → magnetic field")
    axes[1].set_ylabel(r"$\int |B_3|\,dV$")
    axes[1].legend(fontsize=8)

    history = loaded["coupled_gpu1"]["history"]
    axes[2].plot(history["time"], history["laser_E"], label="deposited laser E")
    axes[2].plot(
        history["time"], history["chain_E"]-history["chain_E"][0],
        linestyle="--", label=r"$\Delta(E_{matter}+E_{rad})$",
    )
    axes[2].set_title("Coupled energy closure")
    axes[2].set_ylabel("integrated energy")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("time")
        ax.grid(alpha=0.25)
    fig.savefig(plot_dir / "communication_history.png", dpi=180)
    fig.savefig(plot_dir / "communication_history.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results, loaded = load_cases()
    coupled = loaded["coupled_gpu1"]["result"]
    coupled_mpi = loaded["coupled_gpu2"]["result"]
    no_radiation = loaded["no_radiation_gpu1"]["result"]
    no_biermann = loaded["no_biermann_gpu1"]["result"]

    radiation_excess = (
        coupled["radiation_energy_gain"]-no_radiation["radiation_energy_gain"]
    )
    radiation_communication_ok = bool(
        coupled["radiation_energy_gain"] > 1.0e-10
        and radiation_excess > max(5.0*abs(no_radiation["radiation_energy_gain"]),
                                   1.0e-9)
    )
    biermann_generation_ok = bool(
        coupled["final_abs_bz_integral"] > 1.0e-10
        and coupled["biermann_source_time_integral"] > 0.0
        and coupled["bz_antisymmetry"] > 0.25
        and -0.2 < coupled["bz_peak_x1"] < 0.1
        and no_biermann["final_abs_bz_integral"]
        < max(1.0e-12, 1.0e-4*coupled["final_abs_bz_integral"])
    )

    history_comparison = compare_histories(
        loaded["coupled_gpu1"]["history"], loaded["coupled_gpu2"]["history"]
    )
    field_comparisons = {
        output_id: compare_datasets(
            loaded["coupled_gpu1"]["datasets"][output_id],
            loaded["coupled_gpu2"]["datasets"][output_id],
        )
        for output_id in ("fluid", "three_t", "laser")
    }
    mpi_communication_ok = bool(
        coupled_mpi["max_transfers"] > 0.0
        and history_comparison["pass"]
        and all(item["pass"] for item in field_comparisons.values())
    )
    overall = bool(
        all(result["case_ok"] for result in results)
        and radiation_communication_ok
        and biermann_generation_ok
        and mpi_communication_ok
    )

    payload = {
        "overall": overall,
        "radiation_communication_ok": radiation_communication_ok,
        "biermann_generation_ok": biermann_generation_ok,
        "mpi_communication_ok": mpi_communication_ok,
        "radiation_gain_excess_over_uncoupled": radiation_excess,
        "history_mpi_comparison": history_comparison,
        "field_mpi_comparisons": field_comparisons,
        "cases": results,
    }
    (BENCH_DIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (BENCH_DIR / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "# Laser-target radiation/Biermann benchmark",
        "",
        f"Overall result: {'PASS' if overall else 'FAIL'}",
        "",
        f"- Laser → radiation communication: "
        f"{'PASS' if radiation_communication_ok else 'FAIL'}",
        f"- Biermann generation and odd-B3 topology: "
        f"{'PASS' if biermann_generation_ok else 'FAIL'}",
        f"- MPI ray-transfer/decomposition invariance: "
        f"{'PASS' if mpi_communication_ok else 'FAIL'}",
        "",
        "| Case | GPUs | Laser E | Radiation gain | int |B3| dV | "
        "Energy error | Transfers | Result |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result['case']} | {result['gpus']} | "
            f"{result['laser_energy']:.6e} | "
            f"{result['radiation_energy_gain']:.6e} | "
            f"{result['final_abs_bz_integral']:.6e} | "
            f"{result['chain_laser_relative_error']:.3e} | "
            f"{int(result['max_transfers'])} | "
            f"{'PASS' if result['case_ok'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `no_radiation_gpu1` keeps radiation transport but disables local "
            "electron-radiation source exchange.  Excess radiation energy in the "
            "coupled case therefore measures the laser → electron → radiation path.",
            "- `no_biermann_gpu1` preserves the same hydrodynamics, laser heating, and "
            "radiation but sets the Biermann coefficient to zero.",
            "- In this 2D meridional cut, the toroidal field reported in laser-solid "
            "experiments appears as an antisymmetric out-of-plane B3 pair.",
            "",
            "Paper comparison scope:",
            "",
            "The test checks topology, energy communication, ray conservation, and MPI "
            "invariance.  It does not claim an absolute megagauss comparison because "
            "AthenaK currently uses an ideal-gas EOS, constant thermal opacities, no "
            "radiation force, and a code-unit Biermann coefficient.  See SOURCES.md for "
            "the experimental, FLASH, ray-tracing, and numerical-Biermann references.",
        ]
    )
    (BENCH_DIR / "diagnostics.md").write_text("\n".join(lines) + "\n")
    if not args.no_plots:
        make_plots(loaded)

    print(f"Overall result: {'PASS' if overall else 'FAIL'}")
    print(
        f"radiation={radiation_communication_ok} "
        f"biermann={biermann_generation_ok} mpi={mpi_communication_ok}"
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
