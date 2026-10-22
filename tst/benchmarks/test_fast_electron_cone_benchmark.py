"""Independent quadrature checks against the continuous published source fits."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from fast_electron_cone_benchmark import SOURCES, TP_MEV, source_rows, spectrum


@pytest.mark.parametrize("name", SOURCES)
def test_source_number_energy_and_solid_angle(name):
    rows = source_rows(name, ne=24, nt=16, nphi=8)
    e, th, ph, w = rows.T
    assert np.all(e > 0) and np.all(w > 0)
    assert w.sum() == pytest.approx(1.0, abs=3e-15)
    expected = 1.5 if name.startswith("mono") else TP_MEV*quad(
        lambda x: x*spectrum(x), 0.12, 6.37)[0]/quad(spectrum, 0.12, 6.37)[0]
    assert np.dot(w, e)/1000 == pytest.approx(expected, rel=2e-10)
    delta = math.radians(90 if name.endswith("divergent") else 10)
    f = lambda theta: np.sin(theta)*np.exp(-(theta/delta)**4)
    expected = quad(lambda t: t*f(t), 0, math.pi/2)[0]/quad(f, 0, math.pi/2)[0]
    assert np.dot(w, np.deg2rad(th)) == pytest.approx(expected, rel=3e-6)
    for trig in (np.cos, np.sin):
        assert abs(np.dot(w, np.sin(np.deg2rad(th))*trig(np.deg2rad(ph)))) < 1e-15


def test_angular_support_and_reference_mean():
    rows = source_rows("pic_divergent", nt=16)
    assert rows[:, 1].min() > 0 and rows[:, 1].max() < 90
    assert np.dot(rows[:, 1], rows[:, 3]) == pytest.approx(52.16896319, abs=1e-6)


def test_reject_non_axisymmetric_azimuth_count():
    with pytest.raises(ValueError):
        source_rows("pic_divergent", nphi=3)
