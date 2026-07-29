"""Negative tests: invalid 2T/laser/radiation configurations must fail cleanly.

Each case must exit nonzero with a "### FATAL ERROR" message (never a signal such
as SIGSEGV). Codifies the Phase-1 guard fixes (laser-output null-pointer guard,
FOFC x two_temperature guards) and the constructor compatibility matrix.
"""

import os
import re
import subprocess

import pytest

HYDRO_DECK = "../../../inputs/hydro/two_temperature_relax.athinput"
MHD_DECK = "../../../inputs/mhd/two_temperature_dual_energy.athinput"
LASER_DECK = "../../../inputs/mhd/two_temperature_laser.athinput"


def derive_deck(base, out_name, block_insertions=None, extra_blocks=""):
    """Copy a deck, inserting lines right after selected block headers.

    Any existing assignment of an inserted key is removed first (the parser
    lets later duplicates win, so a plain insertion could be overridden).
    """
    block_insertions = block_insertions or {}
    with open(base) as f:
        text = f.read()
    for block, lines in block_insertions.items():
        for line in lines.strip().splitlines():
            key = line.split("=")[0].strip()
            text = re.sub(r"(?m)^\s*" + re.escape(key) + r"\s*=.*\n", "", text)
        pattern = r"(<" + re.escape(block) + r">\n)"
        text = re.sub(pattern, r"\1" + lines.rstrip() + "\n", text, count=1)
    text += "\n" + extra_blocks
    with open(out_name, "w") as f:
        f.write(text)
    return out_name


def run_expect_fatal(inputfile, flags, needle):
    """Run athena expecting a clean fatal error containing `needle`."""
    command = ["./athena", "-i", inputfile] + flags
    proc = subprocess.run(
        command, capture_output=True, text=True, timeout=120)
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, (
        f"expected failure, got exit 0: {command}")
    assert proc.returncode > 0, (
        f"killed by signal {-proc.returncode} (crash, not clean error): {command}")
    assert "FATAL ERROR" in output, f"no FATAL ERROR message: {command}"
    assert needle in output, (
        f"expected '{needle}' in error output of {command}; got:\n{output[-2000:]}")


RADIATION_BLOCK = """
<thermal_radiation>
n_groups = 1
arad = 1.0
c_light = 10.0
group_bound_0 = 0.0
group_bound_1 = 100.0
kappa_transport_0 = 1.0
"""

LASER_OUTPUT_BLOCK = """
<output9>
file_type = tab
variable = laser_dir1
dt = 1.0
"""


def test_guards():
    try:
        # Control: valid deck runs to completion.
        proc = subprocess.run(
            ["./athena", "-i", HYDRO_DECK, "-d", "guard_ctrl"],
            capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, "control run failed"

        # FOFC x two_temperature (hydro).
        deck = derive_deck(HYDRO_DECK, "guard_fofc_h.athinput",
                           {"hydro": "fofc = true"})
        run_expect_fatal(deck, [], "not yet compatible with FOFC")

        # FOFC x two_temperature (MHD, dual energy off).
        deck = derive_deck(MHD_DECK, "guard_fofc_m.athinput",
                           {"mhd": "fofc = true"})
        run_expect_fatal(deck, ["mhd/dual_energy=false"],
                         "not yet compatible with FOFC")

        # FOFC x dual_energy.
        run_expect_fatal(deck, [], "dual_energy is not yet compatible")

        # <thermal_radiation> without two_temperature.
        deck = derive_deck(HYDRO_DECK, "guard_rad_no2t.athinput",
                           extra_blocks=RADIATION_BLOCK)
        run_expect_fatal(deck, ["hydro/two_temperature=false"],
                         "two_temperature")

        # dual_energy without two_temperature.
        run_expect_fatal(MHD_DECK,
                         ["mhd/two_temperature=false", "mhd/dual_energy=true"],
                         "dual_energy")

        # two_temperature with isothermal EOS (iso_sound_speed must live in the
        # deck: the command line can only override parameters already present).
        deck = derive_deck(HYDRO_DECK, "guard_iso.athinput",
                           {"hydro": "iso_sound_speed = 1.0"})
        run_expect_fatal(deck, ["hydro/eos=isothermal"], "two_temperature")

        # Biermann battery on a 1D mesh.
        deck = derive_deck(MHD_DECK, "guard_bier_1d.athinput",
                           {"mhd": "biermann_battery = true"})
        run_expect_fatal(deck, [], "biermann")

        # Laser: refractive model + critical reflection are exclusive.
        deck = derive_deck(LASER_DECK, "guard_refl_refr.athinput",
                           {"laser": "model = refractive\n"
                                     "critical_reflection = true"})
        run_expect_fatal(deck, [], "critical_reflection")

        # Laser: periodic transport on a non-periodic mesh.
        deck = derive_deck(LASER_DECK, "guard_periodic.athinput",
                           {"laser": "periodic_transport = true"})
        run_expect_fatal(deck, [], "periodic")

        # Laser reflection hysteresis is a fractional drop below the last
        # turning density and must remain in the half-open interval [0, 1).
        for label, value in (("negative", "-0.01"), ("unity", "1.0")):
            deck = derive_deck(
                LASER_DECK, f"guard_hysteresis_{label}.athinput",
                {"laser": f"reflection_hysteresis_fraction = {value}"})
            run_expect_fatal(deck, [], "hysteresis fraction")

        # Laser output variable without a <laser> block (Phase-1 null-guard fix;
        # segfaulted before the fix).
        deck = derive_deck(MHD_DECK, "guard_laser_out.athinput",
                           extra_blocks=LASER_OUTPUT_BLOCK)
        run_expect_fatal(deck, [], "Laser output requested")

        # MHD 2T in a shearing box is unvalidated even with dual_energy=false.
        # (Refinement must be off: shearing box + refinement trips a mesh-level
        # guard before the 2T one.)
        deck = derive_deck(MHD_DECK, "guard_sbox.athinput",
                           extra_blocks="<shearing_box>\nqshear = 1.5\n"
                                        "omega0 = 1.0\n")
        run_expect_fatal(deck,
                         ["mhd/dual_energy=false", "mesh_refinement/refinement=none"],
                         "not yet validated")
    finally:
        for name in os.listdir("."):
            if name.startswith("guard_"):
                subprocess.run(["rm", "-rf", name])
