"""Run the physical comparisons with a GPU-enabled AthenaK build."""

from test_suite.nr.test_nr_flash48_models_cpu import (  # noqa: F401
    test_exchange_invalid_input, test_laser_overdense_launch,
    test_radiation_bath_equilibrium, test_radiation_bath_invalid_input,
    test_radiation_bath_restart, test_radiation_temperature_boundary,
    test_state_dependent_exchange,
)
