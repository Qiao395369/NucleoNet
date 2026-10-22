"""Deterministic correlation references and bounded statistical integration tests."""

import numpy as np
import pytest
from vmcnet.mcmc import statistics


def test_alternating_per_chain_autocorrelation():
    samples = np.tile([[1., 2.], [-1., -2.]], (100, 1))
    actual = statistics.per_chain_autocorr_fast(samples)
    np.testing.assert_allclose(actual, np.tile([[1., 1.], [-1., -1.]], (100, 1)), atol=1e-12)


def test_multichain_correlation_and_variance_reference():
    samples = np.tile([[1., 2., 3.], [-1., -2., -3.]], (10, 1))
    autocorr, variance = statistics.multi_chain_autocorr_and_variance(samples)
    np.testing.assert_allclose(autocorr, np.tile([1., -1.-2/19], 10), atol=1e-12)
    np.testing.assert_allclose(variance, 14/3)


def test_exponential_tau_and_cutoff():
    curve = 0.9**np.arange(1000)
    np.testing.assert_allclose(statistics.tau(curve), 19., rtol=1e-12)
    samples = np.tile([[1., 2.], [-1., -2.]], (10, 1))
    assert statistics.per_chain_autocorr_fast(samples, cutoff=7).shape == (7, 2)


@pytest.mark.slow
@pytest.mark.parametrize("decay,expected_tau", [(0., 1.), (0.8, 9.)])
def test_independent_and_correlated_samples(decay, expected_tau):
    rng = np.random.default_rng(2207)
    noise = rng.normal(size=(100_000, 4))
    samples = np.empty_like(noise)
    samples[0] = noise[0]
    for i in range(1, len(samples)):
        samples[i] = decay*samples[i-1]+noise[i]
    samples = samples[1000:]
    corr, variance = statistics.multi_chain_autocorr_and_variance(samples, cutoff=200)
    np.testing.assert_allclose(corr[:10], decay**np.arange(10), atol=0.015)
    np.testing.assert_allclose(variance, 1/(1-decay**2), rtol=0.035)
    np.testing.assert_allclose(statistics.tau(corr), expected_tau, rtol=0.1)


@pytest.mark.parametrize("repeat", [False, True])
def test_summary_preserves_geometry_and_cut(repeat):
    rng = np.random.default_rng(9)
    samples = rng.normal(size=(400, 8))
    samples[:, 4:] += 20.
    summary = statistics.get_stats_summary(samples, 4, 2, repeat, 10)
    expected = samples[10:].mean() if repeat else [samples[10:, :4].mean(), samples[10:, 4:].mean()]
    np.testing.assert_allclose(summary["average"], expected)
    assert np.isfinite(summary["std_err"]).all()
