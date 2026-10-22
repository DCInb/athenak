#!/usr/bin/env python3
"""Make a mixed-level mesh that exercises collision AMR GPU capacity."""

import argparse
import itertools
import json
from pathlib import Path


def prepare(source, destination, ranks=1):
    """Construct a mixed-level capacity probe without changing the physics."""
    source, destination = Path(source), Path(destination)
    if ranks < 1:
        raise ValueError("ranks must be positive")
    if source.resolve() == destination.resolve():
        raise ValueError("the capacity input must be separate from the production input")
    text = source.read_text()
    sections = {}
    section = None
    lines = text.splitlines(keepends=True)
    for line in lines:
        content = line.split("#", 1)[0].strip()
        if content.startswith("<") and content.endswith(">"):
            section = content[1:-1]
            sections.setdefault(section, {})
        elif section is not None and "=" in content:
            key, value = (part.strip() for part in content.split("=", 1))
            sections[section][key] = value
    mesh, block, refinement = (sections[name] for name in
                               ("mesh", "meshblock", "mesh_refinement"))
    roots = []
    bounds = []
    for axis in (1, 2, 3):
        cells, block_cells = int(mesh[f"nx{axis}"]), int(block[f"nx{axis}"])
        if cells % block_cells:
            raise ValueError("root mesh must be divisible into whole meshblocks")
        roots.append(cells // block_cells)
        bounds.append((float(mesh[f"x{axis}min"]), float(mesh[f"x{axis}max"])))
    if roots == [1, 1, 1]:
        return prepare_single_root(lines, refinement, bounds, destination, ranks)
    if any(name.startswith("refined_region") for name in sections):
        raise ValueError("source already contains a static refined region")
    if roots != [6, 3, 3] or int(refinement["num_levels"]) != 2:
        raise ValueError("the 425-leaf check requires 6 x 3 x 3 roots and two levels")
    if int(refinement["max_nmb_per_rank"]) != 432:
        raise ValueError("the production input must reserve all 432 possible leaves")
    if ranks != 1:
        raise ValueError("the legacy 425-leaf check uses one rank")
    if refinement["refinement"] != "adaptive":
        raise ValueError("adaptive mode is required to preallocate all 432 blocks")
    if int(refinement.get("refinement_interval", "5")) <= 1:
        raise ValueError("refinement_interval must keep the mesh unchanged for one cycle")
    additions = ["\n# Capacity check: retain one coarse corner and refine the other 53 roots.\n",
                 "# Mixed levels exercise the full high-order FGMRES workspace.\n",
                 "# Keep adaptive mode for 432-block preallocation; run only one cycle.\n"]
    for slab in range(3):
        additions.append(f"\n<refined_region{slab+1}>\nlevel = 1\n")
        for axis, ((lower, upper), count) in enumerate(zip(bounds, roots)):
            if axis == slab:
                width = (upper - lower) / count
                # The mesh builder rounds to parent blocks. Stay inside the
                # second parent, avoiding roundoff overlap with the first one.
                lower += width * (1.0 + 1.0e-9)
            additions.append(f"x{axis+1}min = {lower:.17g}\n")
            additions.append(f"x{axis+1}max = {upper:.17g}\n")
    destination.write_text("".join(lines) + "".join(additions))
    return {"input": str(destination), "root_blocks": 54, "coarse_leaves": 1,
            "fine_leaves": 424, "total_leaves": 425, "reserved_blocks": 432}


def prepare_single_root(lines, refinement, bounds, destination, ranks):
    """Use adjacent uniform levels, retaining at least one coarse leaf for FGMRES."""
    capacity = int(refinement["max_nmb_per_rank"])
    levels = int(refinement["num_levels"])
    if refinement["refinement"] != "adaptive":
        raise ValueError("adaptive mode is required for capacity preallocation")
    if int(refinement.get("refinement_interval", "5")) <= 1:
        raise ValueError("refinement_interval must keep the mesh unchanged for one cycle")
    target = min(capacity*ranks, 8**(levels-1))
    if levels < 3 or target < 15:
        raise ValueError("a mixed single-root mesh needs three levels and at least 15 leaves")
    level = 1
    while 8**(level+1) < target:
        level += 1
    coarse = 8**level
    refined = min(coarse-1, (target-coarse)//7)
    count = 2**level
    parents = list(itertools.product(range(count), repeat=3))

    def morton(location):
        return sum(((location[a] >> b) & 1) << (3*b+a)
                   for a in range(3) for b in range(level))

    parents.sort(key=morton)
    # Spread fine parents throughout Morton order and across MPI ranks.
    parents = [parents[i*coarse//refined] for i in range(refined)]
    kept, skip = [], False
    for line in lines:
        content = line.split("#", 1)[0].strip()
        if content.startswith("<") and content.endswith(">"):
            skip = content[1:-1].startswith("refined_region")
        if not skip:
            kept.append(line)

    def region(name, ref_level, lower, upper):
        return (f"\n<refined_region_{name}>\nlevel = {ref_level}\n" + "".join(
            f"x{a+1}min = {lower[a]:.17g}\nx{a+1}max = {upper[a]:.17g}\n"
            for a in range(3)))

    kept.append("\n# Capacity probe: replaces initial SMR regions; run one cycle.\n")
    kept.append(region("capacity_base", level,
                       [b[0] for b in bounds], [b[1] for b in bounds]))
    for index, parent in enumerate(parents):
        center = [lo+(q+0.5)*(hi-lo)/count for q, (lo, hi) in zip(parent, bounds)]
        kept.append(region(f"capacity_{index}", level+1, center, center))
    destination.write_text("".join(kept))
    return {"input": str(destination), "root_blocks": 1, "ranks": ranks,
            "coarse_level": level, "fine_level": level+1,
            "coarse_leaves": coarse-refined, "fine_leaves": 8*refined,
            "total_leaves": coarse+7*refined, "reserved_blocks": capacity*ranks,
            "capacity_per_rank": capacity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="production AMR input")
    parser.add_argument("destination", type=Path, help="separate capacity-check input")
    parser.add_argument("--ranks", type=int, default=1,
                        help="MPI rank count for the single-root capacity check")
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.source, arguments.destination, arguments.ranks), indent=2))


if __name__ == "__main__":
    main()
