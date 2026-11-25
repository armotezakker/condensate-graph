"""Shared pytest settings.

Tests marked ``slow`` run by default on a local machine. In continuous
integration (the ``CI`` environment variable is set) they are skipped unless
``RUN_SLOW=1`` is set or pytest is called with ``--run-slow``.
"""
import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked slow")


def pytest_collection_modifyitems(config, items):
    in_ci = os.environ.get("CI", "").lower() not in ("", "0", "false")
    run_slow = config.getoption("--run-slow") or os.environ.get("RUN_SLOW") == "1"
    if not in_ci or run_slow:
        return
    skip = pytest.mark.skip(reason="slow test; set RUN_SLOW=1 or pass --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
