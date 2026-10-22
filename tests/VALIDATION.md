# Release validation

Date: 2026-10-22.

## Automated tests

Executed with Python 3.10.18, JAX/jaxlib 0.4.34, NumPy 1.26.4, and PySCF 2.10.0.
The test runner uses CPU, float64, and two virtual CPU devices.

```bash
python -B tests/run_tests.py -q --run-slow \
  --junitxml=.artifacts/release-results.xml
```

Result: **110 passed, 56 warnings in 47.11s**. The warnings are JAX `tree_map`
deprecation warnings. The local execution log is
`.artifacts/release-validation.log`; generated test files are excluded from Git
and release packages.

Coverage includes actual small H2 training and evaluation, checkpoint replay,
model symmetries, kinetic and potential energies, MCMC sampling, SPRING updates,
pretraining losses, statistics, configuration parsing, and CLI help.

## Documentation and packaging

Both N2 command examples in the main README were parsed with the production
configuration parser. The single-geometry example has one training/evaluation
geometry and 4096 training chains; the six-geometry example has six
training/evaluation geometries and 684 chains per geometry. Atom counts and spin
allocations were checked. Local Markdown links resolve.

All shell files and the temporary model parameter-counting feature have been
removed. The source archive and wheel are rebuilt from the updated source. The
source archive contains documentation, tests, and license notices. The wheel
contains the Python package, console entry points, and license notices. Generated
results, caches, and test artifacts are excluded.

The local packaging check installs the wheel into a temporary virtual environment
with `--system-site-packages`, reusing the existing runtime dependencies, then
checks package imports, console help, and `pip check`. This is not a fresh download
and installation of all transitive dependencies.

## Limits

The inherited real-network kinetic comparison records and ignores coordinate
NaNs in Hessian/old-Laplacian references and compares finite forward-Laplacian
results. Other finite-value assertions remain active. Passing this comparison
does not validate the reference paths that return NaNs.

The pretraining optimizer test uses analytic reference orbitals; the small
training integration tests disable HF pretraining. Virtual CPU devices exercise
distribution interfaces, not actual multi-GPU execution.

No GPU/CUDA execution, performance measurement, or full-length N2 optimization
was performed for this update. Short integration runs verify execution and file
formats, not converged molecular energies or reliable statistical uncertainties.
