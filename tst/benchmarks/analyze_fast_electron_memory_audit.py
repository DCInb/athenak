#!/usr/bin/env python3
"""Compare archived two-GPU particle memory, sampling, and reclamation evidence.

This reads existing measurements; it does not launch or modify a simulation.
The allocation/live-count amplification is explicitly a nonsimultaneous proxy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKET_BYTES = 104
SOURCES = ["mono_collimated", "pic_collimated", "pic_divergent"]
NAMES = ["Monoenergetic", "PIC spectrum, collimated", "PIC spectrum, divergent"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--convergence", type=Path, default=repo / "docs/data/fast_electron_memory_convergence_2026-10-06")
    parser.add_argument("--reclaim", type=Path, default=repo / "docs/data/fast_electron_reclaim_2026-10-06")
    parser.add_argument("--output", type=Path, default=repo / "docs/data/fast_electron_memory_audit_2026-10-06")
    args = parser.parse_args()
    out = args.output.resolve()
    assert out not in (args.convergence.resolve(), args.reclaim.resolve())
    out.mkdir(parents=True, exist_ok=True)
    verified = {}
    manifests = {}
    for root in (args.convergence, args.reclaim):
        manifests[root] = {name: sha for sha, name in
                           (line.split("  ", 1) for line in (root / "SHA256SUMS").read_text().splitlines())}

    def read(root, relative):
        path = root / relative
        sha = digest(path)
        assert manifests[root][str(relative)] == sha, path
        verified[str(path.resolve().relative_to(repo))] = sha
        return json.loads(path.read_text())

    study = read(args.convergence, Path("memory_convergence.json"))
    releases = read(args.reclaim, Path("allocation_releases.json"))
    repeat_comparisons = read(args.reclaim, Path("comparisons.json"))
    validation = read(args.reclaim, Path("validation.json"))
    assert len(study["rows"]) == 28 and len(study["memory_only"]) == 4
    run_rows, rank_rows, empty_rows, snapshots = [], [], [], []
    for case in study["rows"] + study["memory_only"]:
        cells = case["nx"] ** 3
        ranks = case["memory"]["ranks"]
        assert [rank["rank"] for rank in ranks] == [0, 1]
        live = case["peak_live_packets_at_macro_steps"]
        for rank in ranks:
            total = rank["cuda_peak_bytes"]
            packets = rank["packet_buffers_at_cuda_peak_bytes"]
            templates = rank["source_templates_at_cuda_peak_bytes"]
            mesh = rank["mesh_and_field_at_cuda_peak_bytes"]
            runtime = rank["kokkos_runtime_at_cuda_peak_bytes"]
            assert packets + templates + mesh + runtime == total
            assert packets % PACKET_BYTES == 0
            matches = list(args.convergence.glob(f"*-runs/{case['label']}/memory-rank{rank['rank']}.json"))
            assert len(matches) == 1, (case["label"], matches)
            raw = read(args.convergence, matches[0].relative_to(args.convergence))
            assert raw["duplicate_allocations"] == 0
            cuda = next(space for space in raw["spaces"] if space["space"] == "Cuda")
            assert sum(a["bytes"] for a in cuda["at_peak"]) == total == cuda["peak_bytes"]
            allocations = [a for a in cuda["at_peak"] if "packet" in a["label"]]
            assert sum(a["bytes"] for a in allocations) == packets
            snapshots.append(dict(label=case["label"], rank=rank["rank"], allocations=allocations))
            row = dict(label=case["label"], source=case["source"], nx=case["nx"],
                       rank=rank["rank"], active_cells=cells // 2,
                       packet_buffers_bytes=packets, source_templates_bytes=templates,
                       mesh_field_bytes=mesh, runtime_bytes=runtime, cuda_peak_bytes=total,
                       particle_to_mesh_ratio=(packets + templates) / mesh,
                       particle_percent_of_tracked_cuda=100 * (packets + templates) / total,
                       mesh_bytes_per_active_cell=mesh / (cells // 2),
                       host_peak_rss_bytes=rank["host_max_rss_bytes"])
            (rank_rows if live else empty_rows).append(row)
        if not live:
            continue
        worst = max(ranks, key=lambda rank: rank["cuda_peak_bytes"])
        templates_per_batch = (48 if case["source"] == "mono_collimated" else 1152) * case["source_points"]
        batch = templates_per_batch * case["packets_per_bin"]
        assert case["emitted_packets"] % batch == 0
        run_rows.append(dict(
            label=case["label"], source=case["source"], nx=case["nx"],
            packets_per_bin=case["packets_per_bin"], source_points=case["source_points"], seed=case["seed"],
            global_active_cells=cells, global_observed_peak_live=live,
            global_mean_observed_peak_ppc=live / cells,
            max_rank_cuda_peak_GiB=worst["cuda_peak_bytes"] / 2**30,
            min_rank_cuda_peak_GiB=min(r["cuda_peak_bytes"] for r in ranks) / 2**30,
            worst_rank_particle_MiB=(worst["packet_buffers_at_cuda_peak_bytes"] + worst["source_templates_at_cuda_peak_bytes"]) / 2**20,
            worst_rank_mesh_MiB=worst["mesh_and_field_at_cuda_peak_bytes"] / 2**20,
            worst_rank_particle_to_mesh_ratio=(worst["packet_buffers_at_cuda_peak_bytes"] + worst["source_templates_at_cuda_peak_bytes"]) / worst["mesh_and_field_at_cuda_peak_bytes"],
            nonsimultaneous_packet_amplification_proxy=sum(r["packet_buffers_at_cuda_peak_bytes"] for r in ranks) / (PACKET_BYTES * live),
            rank_peak_ratio=worst["cuda_peak_bytes"] / min(r["cuda_peak_bytes"] for r in ranks),
            emitted_packets=case["emitted_packets"], accepted_emission_batches=case["emitted_packets"] // batch,
            central_deposition_percent=case["central_percent"], loop_s=case["loop_s"],
            max_relative_ledger_error=case["max_relative_ledger_error"]))

    release_rows = []
    for case in releases:
        for event in case["verified_releases"]:
            before, after = event["reserved_before"], event["reserved_after"]
            assert (before - after) * PACKET_BYTES == event["released_bytes"]
            live = event["live_packets"]
            release_rows.append(dict(source=case["source"], rank=case["rank"], **{
                k: v for k, v in event.items() if k != "rank"},
                aggregate_live_fraction_before=live / before,
                aggregate_live_fraction_after=live / after if after else None))
    assert len(release_rows) == validation["verified_reclamation_events"] == 6
    for c in repeat_comparisons:
        assert c["tracked_cuda_peak_bytes_retain"] == c["tracked_cuda_peak_bytes_repeat"] == c["tracked_cuda_peak_bytes_reclaim"]

    convergence = [dict(c, deposition_L1_percent=100 * c["deposition_L1_over_injected"])
                   for c in study["successive_refinements"]]
    definitions = {
        "global_mean_observed_peak_ppc": "Maximum global live count seen at fluid-step diagnostics / global active mesh cells. No ghosts. This misses possible peaks between diagnostics; it is not local or effective PPC.",
        "particle_to_mesh_ratio": "Packet buffers plus immutable source templates divided by mesh/field/halo/deposition/communication arrays in the same rank's CUDA peak snapshot. Labeled Kokkos runtime excluded from both numerator and denominator.",
        "nonsimultaneous_packet_amplification_proxy": "Sum of both ranks' individual peak-snapshot packet-buffer bytes / (104 * maximum observed global live count). Rank peaks and particle count need not be simultaneous; this is NOT exact buffer occupancy, a leak estimate, or a measured bytes/live-particle value.",
        "aggregate_live_fraction": "Live packet count / summed capacity of live, scratch, two pending queues, and export buffer at the same reclamation event. Excludes source templates; scratch is legitimate workspace, so this is not occupancy of the live array alone.",
        "rank_peak_ratio": "Larger/smaller per-rank CUDA peak. Peaks need not be simultaneous; measures capacity pressure disparity, not instantaneous particle imbalance.",
        "cuda_peak": "Allocation-event tracked Kokkos CUDA bytes, including temporary overlap. Excludes driver/context and non-Kokkos allocations; nvidia-smi sampled once/second may miss growth peaks.",
        "accepted_emission_batches": "Total emitted macro-packets / (spectrum rows * spatial source points * packets_per_bin). These inputs have no excluded source points. Counts accepted emission intervals, not rejected timestep retries.",
    }
    result = dict(date="2026-10-06", gpus_per_run=2, packet_bytes=PACKET_BYTES,
                  definitions=definitions, verified_input_sha256=verified,
                  source_sha256={str(p.relative_to(repo)): digest(p) for p in
                      [Path(__file__).resolve(),
                       repo / "src/fast_electrons/fast_electrons.hpp",
                       repo / "src/fast_electrons/fast_electrons.cpp",
                       repo / "src/fast_electrons/fast_electron_mesh.cpp",
                       repo / "src/fast_electrons/fast_electron_transport.cpp",
                       repo / "src/mesh/build_tree.cpp",
                       repo / "src/mesh/mesh_refinement.cpp",
                       repo / "docs/FAST_ELECTRON_MEMORY_AUDIT.md"]},
                  runs=run_rows, per_rank=rank_rows, empty_mesh=empty_rows,
                  reclamation=release_rows, convergence=convergence,
                  seed_comparisons=study["seed_comparisons"], peak_allocations=snapshots,
                  prior_reclamation_validation=validation,
                  checks=dict(verified_input_files=len(verified), physical_runs=len(run_rows),
                              empty_probes=len(empty_rows) // 2, verified_release_events=len(release_rows),
                              peak_snapshot_sums_checked=len(snapshots), new_simulations=0))
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, rows in [("runs", run_rows), ("per_rank", rank_rows), ("empty_mesh", empty_rows),
                       ("reclamation", release_rows), ("convergence", convergence)]:
        write_csv(out / f"{name}.csv", rows)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), layout="constrained")
    baseline = [next(r for r in run_rows if r["source"] == s and r["nx"] == 16 and
                     r["packets_per_bin"] == 1 and r["source_points"] == 64 and r["seed"] == 12345)
                for s in SOURCES]
    ax = axes[0, 0]
    x = np.arange(3)
    ax.bar(x - .18, [r["worst_rank_particle_MiB"] for r in baseline], .36, label="Particle buffers + templates", color="#0072B2")
    ax.bar(x + .18, [r["worst_rank_mesh_MiB"] for r in baseline], .36, label="Mesh + fields + workspace", color="#999999")
    for i, r in enumerate(baseline):
        ax.text(i - .18, r["worst_rank_particle_MiB"] * 1.17, f'{r["worst_rank_particle_to_mesh_ratio"]:.1f}:1', ha="center", fontsize=10)
    ax.set(yscale="log", ylabel="MiB on the GPU with the larger peak", title="A. Memory at the CUDA peak, 16³ mesh", ylim=(3, 50000))
    ax.set_xticks(x, ["Mono", "PIC collimated", "PIC divergent"])
    ax.legend(fontsize=8, loc="upper left")

    for s, name, color in zip(SOURCES, NAMES, COLORS):
        seq = sorted([r for r in run_rows if r["source"] == s and r["packets_per_bin"] == 1
                      and r["source_points"] == 64 and r["seed"] == 12345], key=lambda r: r["nx"])
        ns = [r["nx"] for r in seq]
        axes[0, 1].plot(ns, [r["max_rank_cuda_peak_GiB"] for r in seq], "o-", color=color, label=name)
        axes[0, 1].fill_between(ns, [r["min_rank_cuda_peak_GiB"] for r in seq],
                               [r["max_rank_cuda_peak_GiB"] for r in seq], color=color, alpha=.14)
        axes[1, 0].plot(ns, [r["global_mean_observed_peak_ppc"] for r in seq], "o-", color=color, label=name)
    axes[0, 1].axhline(40, ls="--", color="#777777", lw=1)
    axes[0, 1].text(63, 41, "40 GiB nominal; context excluded", ha="right", fontsize=8)
    axes[0, 1].set(yscale="log", ylim=(.15, 60), xlabel="Cells per dimension", ylabel="Tracked CUDA peak [GiB/GPU]",
                   title="B. Peak memory rises with mesh refinement")
    axes[0, 1].legend(fontsize=8, loc="lower right")
    axes[1, 0].set(yscale="log", xlabel="Cells per dimension", ylabel="Observed global mean peak PPC",
                   title="C. Mean PPC falls while total particles grow")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.set_xticks([16, 24, 32, 64])
    ax = axes[1, 1]
    for i, (s, color) in enumerate(zip(SOURCES, COLORS)):
        pairs = [("packet_samples", f"mesh16-p4-{s}"), ("source_points", f"mesh16-source256-{s}"), ("mesh", f"mesh32-p1-{s}")]
        values = [next(c["deposition_L1_percent"] for c in convergence if c["family"] == family and c["fine"] == fine)
                  for family, fine in pairs]
        bars = ax.bar(x + (i - 1) * .25, values, .25, color=color)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value*1.12, f"{value:.2f}", ha="center", fontsize=8)
    ax.set(yscale="log", ylim=(.1, 150), ylabel="Deposition L1 difference / injected energy [%]",
           title="D. Extra packet samples do not resolve space")
    ax.set_xticks(x, ["Samples 2 → 4", "Source points\n144 → 256", "Mesh\n24³ → 32³"])
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Fast-electron memory audit: two A100 GPUs, 1 ps pulse, 1.5 ps endpoint\n"
                 "Memory ratios are simultaneous snapshots; PPC is a fluid-step-sampled domain average", fontsize=13)
    for ext in ("png", "pdf"):
        options = {"metadata": {"CreationDate": None, "ModDate": None}} if ext == "pdf" else {}
        fig.savefig(out / f"memory_audit.{ext}", dpi=170, **options)
    plt.close(fig)
    artifacts = [out / name for name in ("audit.json", "runs.csv", "per_rank.csv",
                 "empty_mesh.csv", "reclamation.csv", "convergence.csv",
                 "memory_audit.png", "memory_audit.pdf")]
    (out / "SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.name}\n" for path in sorted(artifacts)))
    print(json.dumps(result["checks"], indent=2))


if __name__ == "__main__":
    main()
