#!/usr/bin/env bash
# Copy the reference data without regenerating or modifying DCI_3D's audited tables.
set -euo pipefail
case_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source_dir=${TABLE_SOURCE_DIR:-"${case_dir}/material_tables"}
[[ -d "$source_dir" ]] || source_dir="${case_dir}/../DCI_3D/material_tables"
destination=${1:-"${case_dir}/material_tables"}
mkdir -p -- "$destination"
while read -r expected filename; do
  if [[ -f "${destination}/${filename}" ]]; then
    actual=$(sha256sum -- "${destination}/${filename}")
    [[ ${actual%% *} == "$expected" ]] || {
      printf 'Table checksum mismatch: %s\n' "${destination}/${filename}" >&2
      exit 1
    }
    continue
  fi
  actual=$(sha256sum -- "${source_dir}/${filename}")
  [[ ${actual%% *} == "$expected" ]] || {
    printf 'Reference checksum mismatch: %s\n' "${source_dir}/${filename}" >&2
    exit 1
  }
  cp -- "${source_dir}/${filename}" "${destination}/${filename}"
done < "${case_dir}/material_tables.sha256"
cd -- "$destination"
sha256sum --check "${case_dir}/material_tables.sha256"
