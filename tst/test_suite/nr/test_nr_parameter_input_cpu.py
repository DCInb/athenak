"""Large AMR inputs and expanded restart headers must remain readable."""

import os
from pathlib import Path
import re
import subprocess

import pytest


def parse_input(path, restart=False):
    executable = Path(os.environ.get("ATHENA_PARAMETER_TEST_EXE", "./athena")).resolve()
    command = [str(executable), "-r" if restart else "-i", str(path), "-n"]
    ranks = int(os.environ.get("ATHENA_PARAMETER_TEST_RANKS", "1"))
    if ranks > 1:
        command = ["mpirun", "--oversubscribe", "-n", str(ranks), *command]
    return subprocess.run(command, capture_output=True, text=True, timeout=60,
                          check=False)


@pytest.mark.parametrize("restart", (False, True))
def test_large_amr_parameter_header(tmp_path, restart):
    # Real region parameters survive ParameterDump, unlike comment-only padding.
    header = "".join(
        f"<refined_region{i}>\nlevel = 1\nx1min = 0.1\nx1max = 0.9\n"
        "x2min = 0.1\nx2max = 0.9\nx3min = 0.1\nx3max = 0.9\n"
        for i in range(640))
    assert len(header) > 40*1024
    if restart:
        # Split the terminator across two 4096-byte reads and append binary data.
        padding = (4093-len(header)) % 4096
        if padding < 2:
            padding += 4096
        header += "#" + "x"*(padding-2) + "\n"
        assert len(header) % 4096 == 4093
        payload = header.encode() + b"<par_end>\n" + bytes(range(256))
    else:
        payload = header.encode()
    path = tmp_path/("large.rst" if restart else "large.athinput")
    path.write_bytes(payload)
    result = parse_input(path, restart)
    assert result.returncode == 0, result.stdout+result.stderr
    assert len(re.findall(r"<refined_region\d+>", result.stdout)) == 640
    assert "<refined_region639>" in result.stdout
    assert "<par_end>" in result.stdout


def test_unterminated_oversized_parameter_header_is_rejected(tmp_path):
    path = tmp_path/"oversized.athinput"
    path.write_text("#" + "x"*(1024*1024+4096))
    result = parse_input(path)
    assert result.returncode != 0
    assert "Parameter header exceeds 1 MiB" in result.stdout+result.stderr
