"""Local collection policy; never imports either legacy test directory."""

from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", "--run_slow", action="store_true", default=False,
                     help="Run real-network/training/statistical integration tests")


def pytest_configure(config):
    # Also contain pytest temporary files when invoking pytest directly.
    if config.option.basetemp is None:
        config.option.basetemp = str(Path(__file__).parent / ".artifacts" / "pytest-tmp")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-slow"):
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="enable with --run-slow"))


@pytest.fixture(scope="session", autouse=True)
def double_precision():
    import jax
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)
