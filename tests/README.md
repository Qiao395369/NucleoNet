# Test suite

Run from the repository root with Python 3.10 and the project dependencies:

```bash
python -m pip install -e '.[testing]'
python -B tests/run_tests.py -q
python -B tests/run_tests.py -q --run-slow
```

To select a module or a test name:

```bash
python -B tests/run_tests.py -q test_spring.py
python -B tests/run_tests.py -q --run-slow -k checkpoint
```

The launcher uses the current Python interpreter. It defaults to CPU, float64,
and two virtual CPU devices; existing `JAX_PLATFORMS` and `XLA_FLAGS` are retained.
For one CPU device:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=1 python -B tests/run_tests.py -q
```

Generated files are kept under `tests/.artifacts/` and ignored by Git. The
`.artifacts/pytest-tmp` directory is recreated by pytest on each run; do not store
personal files there. The launcher disables bytecode, pytest caching, and
third-party plugin autoloading.

## Coverage

| Module | Coverage |
| --- | --- |
| `test_config.py` | Defaults, independent configs, CLI tuple fields, unknown flags, presets, reload precedence |
| `test_physics.py` | Coulomb potentials, Gaussian/atomic analytic kinetic energies, local-energy clipping, electron initialization |
| `test_models.py` | NucleoNet and ablation, antisymmetry, translations, equal-charge nuclear permutations, descriptors, envelopes, Jastrows, determinant arithmetic, kinetic energy |
| `test_mcmc.py` | Acceptance rules, geometry axis, amplitude cache, adaptive steps, scans, Gaussian sampling |
| `test_spring.py` | Independent linear-system reference, damping/history, update bounds, schedules, actual derivatives and pmap reductions |
| `test_statistics.py` | Autocorrelation, variance, integrated autocorrelation time, independent/AR(1) samples, geometry summaries |
| `test_io_checkpoint.py` | Checkpoint state round trips, async writers, best checkpoints, metrics, JSON and unique paths |
| `test_training.py` | Gradient accumulation, optimizer updates, logs and checkpoint replay |
| `test_runner_integration.py` | Actual small H2 training/evaluation, output files, resume-versus-continuous training |
| `test_pretrain.py` | Slater blocks, subspace losses, determinant axis, off-block penalty, actual optimizer update |
| `test_cli.py` | Help exits without parsing a training configuration or starting a calculation |

## Verification limits

The suite uses `gq.rescale_input="all"`. The real-network kinetic comparison
records and ignores coordinate-derivative NaNs in reference paths (Hessian and
old Laplacian), and compares finite forward-Laplacian results. This is an inherited
test policy, not evidence that the reference derivatives are numerically valid.
Other finite-value assertions, including parameter gradients and training output,
remain active. The JUnit report records ignored paths and finite comparison counts.

The HF optimizer test uses analytic reference orbitals and does not run a PySCF
SCF solve. Real molecule integration tests disable HF pretraining and use small
batches and networks. Virtual CPU devices check distribution interfaces, not
actual multi-GPU execution. Long-run convergence, CUDA operation, and manuscript
energy precision require separate scientific validation.

See [VALIDATION.md](VALIDATION.md) for the release validation results.
