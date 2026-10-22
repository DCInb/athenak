"""Reject a default executable that predates the Athena or DCI source files."""

from pathlib import Path
import sys


def newer_source(executable, repo, case):
    built_at = executable.stat().st_mtime_ns
    sources = [repo / "CMakeLists.txt", repo / "config.hpp.in",
               case / "dci_integrate.cpp", case / "dci_amr.hpp"]
    for directory in (repo / "src", repo / "cmake"):
        sources.extend(
            path for path in directory.rglob("*")
            if path.suffix in {".cpp", ".hpp", ".h", ".cmake", ".in"}
            or path.name == "CMakeLists.txt"
        )
    return next((path for path in sources
                 if path.is_file() and path.stat().st_mtime_ns > built_at), None)


def main():
    case = Path(__file__).resolve().parent
    executable = Path(sys.argv[1]).resolve()
    stale = newer_source(executable, case.parent, case)
    if stale is not None:
        print(
            f"Default executable is older than source: {executable}\n"
            f"Newer source: {stale}\n"
            f"Rebuild with sbatch {case / 'make.slurm'} and wait for it to finish.\n"
            "To deliberately run a separately validated build, set ATHENA_EXE.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
