"""Independent checks of the analytic oracles and dimensional normalization."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from fast_electron_bai_benchmark import Case, ME, QE, CLIGHT, gyration_exact, oscillation_exact


def test_gyration_against_independent_lorentz_integration():
    times = np.linspace(0., 50., 101)

    def rhs(t, state):
        py, pz = state[2:]
        gamma = math.sqrt(1+(py*py+pz*pz)/100.)
        vy, vz = py/gamma, pz/gamma
        return [vy, vz, vz-1., -vy]

    solution = solve_ivp(rhs, (0., 50.), [0., 0., 10., 0.], t_eval=times,
                         method="DOP853", atol=1e-12, rtol=1e-12)
    assert solution.success
    position, momentum = gyration_exact(times)
    np.testing.assert_allclose(position, solution.y[:2].T, atol=2e-9)
    np.testing.assert_allclose(momentum, solution.y[2:].T, atol=2e-9)


def test_electron_oscillation_against_coupled_lorentz_integration():
    case = Case("oscillation", ppc=64)
    p0 = case.gamma*0.1
    times = np.linspace(0., case.tlim, 101)

    def rhs(t, state):
        px, py, ux, uy = state
        gamma = math.sqrt(1+(px*px+py*py)/case.c**2)
        dpx, dpy = -(py/gamma-uy), px/gamma-ux
        return [dpx, dpy, -case.loading*dpx, -case.loading*dpy]

    solution = solve_ivp(rhs, (0., case.tlim), [0., p0, 0., -case.loading*p0],
                         t_eval=times, method="DOP853", atol=1e-13, rtol=1e-12)
    assert solution.success
    exact = oscillation_exact(times, case)
    np.testing.assert_allclose(exact, solution.y[:2].T, atol=2e-11)
    np.testing.assert_allclose(-case.loading*exact, solution.y[2:].T, atol=2e-11)


@pytest.mark.parametrize("case", [Case("gyration"), Case("bell", c=100, ppc=2),
                                  Case("oscillation", ppc=64)])
def test_units_recover_prescribed_light_speed_and_gyrofrequency(case):
    bunit = math.sqrt(4*math.pi*case.mass_unit/case.time_unit**2)
    assert CLIGHT*case.time_unit == pytest.approx(case.c, rel=1e-14)
    assert QE*bunit*case.time_unit/(ME*CLIGHT) == pytest.approx(case.omega, rel=1e-14)


def test_bell_seed_is_the_growing_eigenmode():
    k, eps = 2*np.pi, 0.1
    # Complex Fourier amplitudes used by seed_particles; negative electron charge
    # and B_x=-1. This independently checks the polarization and drift signs.
    b = np.array([-1., 1.j])
    u = (eps+1j*math.sqrt(1-eps*eps))*b
    rotate = np.array([[0, -1], [1, 0]])
    db = -1j*k*u
    du = -1j*k*b+2*k*rotate@b-2*k*eps*rotate@u
    eigenvalue = k*math.sqrt(1-eps*eps)-1j*k*eps
    np.testing.assert_allclose(db, eigenvalue*b, atol=1e-14)
    np.testing.assert_allclose(du, eigenvalue*u, atol=1e-14)
