"""Run leaf-ownership and regridding checks with distributed ray migration."""

import test_suite.testutils as testutils
from test_suite.nr import test_nr_laser_amr_cpu as amr


def test_run(monkeypatch):
    def run_distributed(inputfile, flags=None, **kwargs):
        return testutils.mpi_run(inputfile, flags=flags, threads=2,
                                 timeout=120, **kwargs)

    monkeypatch.setattr(testutils, "run", run_distributed)
    amr.test_run()
