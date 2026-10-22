# NucleoNet

NucleoNet is a JAX implementation of a nuclear-aware neural wavefunction for molecular variational Monte Carlo (VMC). It supports single-geometry optimization and joint optimization across multiple geometries of the same molecular system, using a shared electron–nucleus backbone, a multideterminant Slater–Jastrow ansatz, forward Laplacians, and the SPRING optimizer.

The Python package is named `vmcnet`. The examples below define molecular coordinates directly on the command line; no external dataset is needed. Pretrained checkpoints and training trajectories are not included.

## Installation

Use **Python 3.10**. Direct runtime dependencies are pinned in [setup.cfg](setup.cfg), including JAX/jaxlib 0.4.34, NumPy 1.26.4, and PySCF 2.10.0. The forward-Laplacian implementation uses JAX internals; keep the pinned JAX version. These dependency pins are not a complete lockfile for transitive dependencies or CUDA libraries.

```bash
git clone https://github.com/Qiao395369/NucleoNet.git
cd NucleoNet
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[testing]'
python -m pip check
```

For NVIDIA GPUs on Linux, install the CUDA 12 extra:

```bash
python -m pip install -e '.[testing,cuda12]'
python -c "import jax; print(jax.__version__); print(jax.devices())"
```

Confirm that JAX lists a GPU before starting a full calculation. Driver and platform requirements are described in the [JAX 0.4.34 installation instructions](https://github.com/jax-ml/jax/blob/jax-v0.4.34/docs/installation.md). CPU execution is suitable for small checks; the examples below are full training configurations.

## Single-geometry N2

Run this command from the repository root. The N–N distance is 2.068 bohr. The network uses one-/two-particle widths of 64/16, four layers, and 16 determinants.

```bash
python -m vmcnet.train.runners \
  --config.logdir=./results \
  --config.subfolder_name=N2 \
  --config.save_to_current_datetime_subfolder=False \
  --config.dtype=float64 \
  --config.problem.ion_pos='(((0.,0.,1.034),(0.,0.,-1.034)),)' \
  --config.problem.ion_charges='(7.,7.)' \
  --config.problem.atoms_symbol='("N","N")' \
  --config.problem.nspins='(7,7)' \
  --config.problem.single_nspins='((5,2),(2,5))' \
  --config.pretrain.nburn=0 \
  --config.pretrain.iterations=4000 \
  --config.vmc.nchains=4096 \
  --config.vmc.nburn=4000 \
  --config.vmc.nepochs=150000 \
  --config.vmc.fwdlap_inner_size=42 \
  --config.gq.type=ef_shrd_sym \
  --config.gq.envelope_type=ds_hz_deg \
  --config.gq.ndet=16 \
  --config.gq.wfn_depth=4 \
  --config.gq.h1=64 \
  --config.gq.h2=16 \
  --config.eval.ion_pos='(((0.,0.,1.034),(0.,0.,-1.034)),)' \
  --config.eval.ion_charges='(7.,7.)' \
  --config.eval.nspins='(7,7)' \
  --config.eval.single_nspins='((5,2),(2,5))' \
  --config.eval.nchains=128 \
  --config.eval.nburn=40000 \
  --config.eval.nepochs=50000
```

The run performs UHF/cc-pVDZ pretraining, VMC optimization, and fixed-parameter energy evaluation. Coordinates are in **bohr** and energies in **hartree**.

## Six-geometry N2

This example jointly trains one model at N–N distances of **1.6, 2.48, 3.36, 4.24, 5.12, and 6.0 bohr**, then evaluates the same six geometries with the optimized parameters held fixed. The network widths are 128/32.

```bash
python -m vmcnet.train.runners \
  --config.logdir=./results \
  --config.subfolder_name=N2_multi_6 \
  --config.save_to_current_datetime_subfolder=False \
  --config.dtype=float64 \
  --config.problem.ion_pos='(
    ((0.,0.,0.8),(0.,0.,-0.8)),
    ((0.,0.,1.24),(0.,0.,-1.24)),
    ((0.,0.,1.68),(0.,0.,-1.68)),
    ((0.,0.,2.12),(0.,0.,-2.12)),
    ((0.,0.,2.56),(0.,0.,-2.56)),
    ((0.,0.,3.0),(0.,0.,-3.0))
  )' \
  --config.problem.ion_charges='(7.,7.)' \
  --config.problem.atoms_symbol='("N","N")' \
  --config.problem.nspins='(7,7)' \
  --config.problem.single_nspins='((5,2),(2,5))' \
  --config.pretrain.nburn=0 \
  --config.pretrain.iterations=4000 \
  --config.vmc.nchains=684 \
  --config.vmc.nburn=4000 \
  --config.vmc.nepochs=150000 \
  --config.vmc.fwdlap_inner_size=21 \
  --config.gq.type=ef_shrd_sym \
  --config.gq.envelope_type=ds_hz_deg \
  --config.gq.ndet=16 \
  --config.gq.wfn_depth=4 \
  --config.gq.h1=128 \
  --config.gq.h2=32 \
  --config.eval.ion_pos='(
    ((0.,0.,0.8),(0.,0.,-0.8)),
    ((0.,0.,1.24),(0.,0.,-1.24)),
    ((0.,0.,1.68),(0.,0.,-1.68)),
    ((0.,0.,2.12),(0.,0.,-2.12)),
    ((0.,0.,2.56),(0.,0.,-2.56)),
    ((0.,0.,3.0),(0.,0.,-3.0))
  )' \
  --config.eval.ion_charges='(7.,7.)' \
  --config.eval.nspins='(7,7)' \
  --config.eval.single_nspins='((5,2),(2,5))' \
  --config.eval.nchains=128 \
  --config.eval.nburn=40000 \
  --config.eval.nepochs=50000
```

`vmc.nchains` is the number of chains **per geometry**. The six-geometry example therefore uses `684 × 6 = 4104` electronic configurations per update. All six geometries share the same model parameters.

To evaluate other bond lengths, change `config.eval.ion_pos` while preserving the atom order, charges, and spin populations. Training and evaluation coordinates are configured separately.

## Configuration and outputs

All fields are defined in [default_config.py](vmcnet/train/default_config.py). The examples use float64, SPRING, 40 MCMC steps per VMC update, and a residual MLP Jastrow. HF pretraining uses Adam at 0.0003 and 20 MCMC steps per update. `pretrain.iterations=0` disables pretraining. For a shorter execution check, reduce the chain counts, burn-in steps, and iteration counts in the command before running it; short trajectories do not give converged energies or reliable error bars.

`single_nspins` specifies the initial spin-up/down electron allocation per nucleus; its column sums must match `nspins`. A positive `vmc.fwdlap_inner_size` must divide `3 × nelectrons`; zero uses the full forward Laplacian.

Outputs are written to `results/N2/` or `results/N2_multi_6/`. Existing directories receive a numerical suffix. Typical files are:

| File | Contents |
| --- | --- |
| `config.json` | Resolved training and evaluation configuration |
| `reload_config.json` | Reload settings |
| `energy.txt`, `variance.txt` | Mean training energy and variance per update |
| `multi_energy.txt` | Mean training energy for each geometry per update |
| `checkpoints/<step>.npz` | Regular checkpoints, every 10,000 updates by default |
| `best_checkpoint.npz` | Checkpoint selected by the running energy/variance metric |
| `eval/multi_energy.txt` | Individual local-energy samples at fixed parameters |
| `eval/statistics.json` | Mean energy, variance, autocorrelation-corrected standard error, and integrated autocorrelation time per geometry |

Training `multi_energy.txt` contains geometry means. Evaluation `eval/multi_energy.txt` contains individual local energies, with shape `(nsteps, ngeometries × nchains)`. Geometry blocks follow the order of `eval.ion_pos`. Statistics are computed automatically from the evaluation samples. The standard error is `sqrt(tau × variance / nsamples)`.

Evaluation immediately after training uses the final parameters in memory. Checkpoints contain parameters, optimizer state, samples, and RNG state, and should only be loaded from trusted sources.

## Reload and resume training

For example, restore the single-geometry N2 run from `results/N2/checkpoints/100000.npz` and continue to a total of 150,000 VMC updates:

```bash
python -m vmcnet.train.runners \
  --reload.logdir=./results/N2 \
  --reload.use_config_file=True \
  --reload.use_checkpoint_file=True \
  --reload.checkpoint_relative_file_path=checkpoints/100000.npz \
  --reload.end_epochs=150000 \
  --reload.new_optimizer_state=False \
  --reload.same_logdir=False \
  --config.subfolder_name=N2_reload \
  --config.pretrain.iterations=0
```

`reload.logdir` must contain the saved `config.json`; the checkpoint path is relative to that directory. The saved configuration supplies the molecular system and network architecture. Parameters, optimizer state, sampling state, and RNG state are restored from the checkpoint. HF pretraining is disabled for this resumed run.

`reload.end_epochs` is the **absolute training endpoint**, not the number of additional updates. With the original output root `./results`, this example writes to `results/N2_reload/`, adding a suffix if needed, and evaluates the resumed model using the saved evaluation settings. To resume the six-geometry example, change `reload.logdir` to `./results/N2_multi_6` and choose a suitable output subfolder. Keep the architecture, molecular identity, precision, and device layout compatible with the checkpoint.

## Tests and packaging

```bash
python -B tests/run_tests.py -q
python -B tests/run_tests.py -q --run-slow
python -m pip install build
python -m build
```

Tests default to CPU, float64, and two virtual CPU devices. See [tests/README.md](tests/README.md) for coverage and [tests/VALIDATION.md](tests/VALIDATION.md) for results and limitations. Small integration tests do not validate GPU operation or long-run energy convergence. The real-network kinetic comparison records and ignores existing coordinate-derivative NaNs in Hessian/old-Laplacian reference paths; production training uses the forward Laplacian.

The source archive includes documentation, tests, and license notices. The wheel installs `vmcnet` and the `vmc-molecule` and `vmc-statistics` console commands. Generated results, caches, and test artifacts are excluded from distributions.

## Code layout and licenses

| Path | Purpose |
| --- | --- |
| `vmcnet/gaoqiao/` | Backbone, descriptors, determinants, envelopes, and Jastrow |
| `vmcnet/train/` | Configuration, HF pretraining, VMC, evaluation, and statistics |
| `vmcnet/mcmc/` | Metropolis sampling and autocorrelation statistics |
| `vmcnet/physics/` | Coulomb potentials, kinetic energy, and forward Laplacian |
| `vmcnet/updates/` | SPRING optimizer and schedules |
| `vmcnet/lapnet_/` | HF/PySCF interfaces and atomic utilities |
| `vmcnet/utils/` | Checkpoints, metrics, and device distribution |

The existing [MIT license](LICENSE) is preserved. Bundled files with Apache-2.0 headers retain that license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).
