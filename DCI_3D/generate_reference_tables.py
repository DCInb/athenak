#!/usr/bin/env python3
"""Generate local AthenaK material tables from the audited 3d_zb archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY = CASE_DIR.parent
DEFAULT_ARCHIVE = REPOSITORY / "3d_zb.zip"
DEFAULT_OUTPUT_DIR = CASE_DIR / "material_tables"
CONVERTER = REPOSITORY / "scripts" / "flash_cn4_to_athenak.py"
ARCHIVE_SHA256 = "952708009c9e3bc00dc645e11c9c0f804614def9c70cc999b78c92f16c8a96cf"
EXPECTED_OUTPUT_SHA256 = {
    "ch.2t_eos": "b29624877c7c90ed1d8c385bef6a7882b106dd8202bf0398301e2dee09faa0d8",
    "he.2t_eos": "aae12f2dde296992ad630094e5755f7f52baa0816c678771e075f4848a9d63d0",
    "ch_20g.opacity": "47ee4b8ab3e7f249e4b7108ab5efbaabeee71bb7dd88cdea59f4b4c64738f94d",
    "he_20g.opacity": "1e0daba15df1a23f5f558663e867dda181886ea53b0dbad119beb6fa1215f420",
}

OPACITY_TABLES = (
    (
        "CH",
        "3d_zb/feos_snop_CH_20g.cn4",
        6.5,
        "ch_20g.opacity",
    ),
    (
        "He",
        "3d_zb/He20g.cn4",
        4.002602,
        "he_20g.opacity",
    ),
)

TWO_TEMPERATURE_TABLES = (
    (
        "CH",
        "3d_zb/C16H1620gPROP.cn4",
        6.5,
        "ch.2t_eos",
    ),
    (
        "He",
        "3d_zb/He_20G_yr23.cn4",
        4.002602,
        "he.2t_eos",
    ),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPOSITORY))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--force", action="store_true", help="replace existing generated tables"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Reference archive not found: {archive}")
    archive_sha = sha256_path(archive)
    if archive_sha != ARCHIVE_SHA256:
        raise RuntimeError(
            f"Reference archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, "
            f"found {archive_sha}"
        )
    if not CONVERTER.is_file():
        raise FileNotFoundError(f"CN4 converter not found: {CONVERTER}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "archive": repository_relative_or_absolute(archive),
        "archive_sha256": archive_sha,
        "converter": repository_relative_or_absolute(CONVERTER),
        "tables": [],
    }
    with zipfile.ZipFile(archive) as source_zip, tempfile.TemporaryDirectory(
        prefix="dci_3d_cn4_"
    ) as temporary:
        temporary_dir = Path(temporary)
        available = set(source_zip.namelist())
        for material, member, abar, output_name in OPACITY_TABLES:
            if member not in available:
                raise RuntimeError(f"Reference archive is missing {member}")
            payload = source_zip.read(member)
            source_path = temporary_dir / Path(member).name
            source_path.write_bytes(payload)
            output_path = output_dir / output_name
            command = [
                sys.executable,
                str(CONVERTER),
                str(source_path),
                "--opacity-output",
                str(output_path),
                "--abar",
                f"{abar:.17g}",
                "--grid-mode",
                "auto",
                "--electron-entropy",
                "auto",
                "--quiet",
            ]
            if args.force:
                command.append("--force")
            subprocess.run(command, cwd=REPOSITORY, check=True)
            output_sha = sha256_path(output_path)
            if output_sha != EXPECTED_OUTPUT_SHA256[output_name]:
                raise RuntimeError(
                    f"Generated {output_name} SHA-256 mismatch: expected "
                    f"{EXPECTED_OUTPUT_SHA256[output_name]}, found {output_sha}"
                )
            manifest["tables"].append(
                {
                    "kind": "opacity",
                    "material": material,
                    "abar": abar,
                    "archive_member": member,
                    "archive_member_sha256": hashlib.sha256(payload).hexdigest(),
                    "output": output_name,
                    "output_sha256": output_sha,
                }
            )

        for material, member, abar, output_name in TWO_TEMPERATURE_TABLES:
            if member not in available:
                raise RuntimeError(f"Reference archive is missing {member}")
            payload = source_zip.read(member)
            source_path = temporary_dir / Path(member).name
            source_path.write_bytes(payload)
            output_path = output_dir / output_name
            command = [
                sys.executable,
                str(CONVERTER),
                str(source_path),
                "--two-temperature-output",
                str(output_path),
                "--abar",
                f"{abar:.17g}",
                "--grid-mode",
                "auto",
                "--electron-entropy",
                "auto",
                "--quiet",
            ]
            if args.force:
                command.append("--force")
            subprocess.run(command, cwd=REPOSITORY, check=True)
            output_sha = sha256_path(output_path)
            if output_sha != EXPECTED_OUTPUT_SHA256[output_name]:
                raise RuntimeError(
                    f"Generated {output_name} SHA-256 mismatch: expected "
                    f"{EXPECTED_OUTPUT_SHA256[output_name]}, found {output_sha}"
                )
            manifest["tables"].append(
                {
                    "kind": "two_temperature_eos",
                    "material": material,
                    "abar": abar,
                    "archive_member": member,
                    "archive_member_sha256": hashlib.sha256(payload).hexdigest(),
                    "output": output_name,
                    "output_sha256": output_sha,
                }
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    count = len(OPACITY_TABLES) + len(TWO_TEMPERATURE_TABLES)
    print(f"Generated {count} material tables in {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
