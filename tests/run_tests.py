"""Run this suite in a fresh process, keeping generated files inside this folder."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    suite = Path(__file__).resolve().parent
    artifacts = suite / ".artifacts"
    (artifacts / "tmp").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(suite.parent),
        TMPDIR=str(artifacts / "tmp"),
        XDG_CACHE_HOME=str(artifacts / "cache"),
        MPLCONFIGDIR=str(artifacts / "matplotlib"),
        JAX_ENABLE_COMPILATION_CACHE="false",
    )
    env.setdefault("JAX_ENABLE_X64", "true")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
    args = [
        sys.executable, "-B", "-m", "pytest", "-c", str(suite / "pytest.ini"),
        "--basetemp", str(artifacts / "pytest-tmp"), *sys.argv[1:],
    ]
    return subprocess.call(args, cwd=suite, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
