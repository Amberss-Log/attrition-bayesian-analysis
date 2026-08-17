"""
metropolis_hastings.py

A from-scratch random-walk Metropolis-Hastings sampler, written to validate the
PyMC/NUTS posterior in 03_bayesian_model.ipynb against a hand-implemented MCMC
method for a simplified single-predictor logistic regression model:

    logit(p) = intercept + beta_overtime * OverTime_Yes

This is intentionally a minimal, dependency-light implementation (numpy only) to
demonstrate the mechanics behind the MCMC/Metropolis-Hastings coursework the
model in this repo builds on — not a general-purpose sampler.
"""

import numpy as np


def log_prior(theta, prior_sd=2.0):
    """
    Log-density of independent Normal(0, prior_sd) priors on [intercept, beta].

    Weakly informative: on this feature's scale, a coefficient of a few units
    already implies a very large effect on attrition odds, so this mildly
    regularizes without ruling out genuinely strong effects.
    """
    intercept, beta = theta
    return (
        -0.5 * (intercept / prior_sd) ** 2
        - 0.5 * (beta / prior_sd) ** 2
    )


def log_likelihood(theta, X, y):
    """
    Log-likelihood of a Bernoulli logistic regression with a single predictor.

    Uses np.logaddexp for numerical stability instead of computing
    sigmoid(x) directly (which can overflow/underflow for large |x|).
    """
    intercept, beta = theta
    logit_p = intercept + beta * X
    log_p = -np.logaddexp(0, -logit_p)         # log(sigmoid(logit_p))
    log_1_minus_p = -np.logaddexp(0, logit_p)  # log(1 - sigmoid(logit_p))
    return np.sum(y * log_p + (1 - y) * log_1_minus_p)


def log_posterior(theta, X, y, prior_sd=2.0):
    """Unnormalized log-posterior = log-likelihood + log-prior."""
    return log_likelihood(theta, X, y) + log_prior(theta, prior_sd=prior_sd)


def metropolis_hastings(X, y, n_iter=20000, proposal_sd=0.05, prior_sd=2.0, seed=42):
    """
    Random-walk Metropolis-Hastings sampler for [intercept, beta].

    Parameters
    ----------
    X, y : array-like
        Predictor (e.g. OverTime_Yes, 0/1) and binary outcome (Attrition_binary).
    n_iter : int
        Number of MCMC iterations (including burn-in — caller should discard
        an initial burn-in window, e.g. the first 5000 draws, before use).
    proposal_sd : float
        Standard deviation of the symmetric Gaussian random-walk proposal.
        ~0.05 gives a healthy acceptance rate (~20-40%) for this model/scale;
        tune this if the acceptance rate printed below is far outside that range.
    prior_sd : float
        Standard deviation of the Normal(0, prior_sd) priors on both parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    samples : np.ndarray, shape (n_iter, 2)
        Columns are [intercept, beta] draws at every iteration (pre burn-in removal).
    acceptance_rate : float
        Fraction of proposals accepted — a basic sampler-health diagnostic.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    theta_current = np.array([0.0, 0.0])
    log_post_current = log_posterior(theta_current, X, y, prior_sd=prior_sd)

    samples = np.zeros((n_iter, 2))
    n_accepted = 0

    for i in range(n_iter):
        theta_proposal = theta_current + rng.normal(0, proposal_sd, size=2)
        log_post_proposal = log_posterior(theta_proposal, X, y, prior_sd=prior_sd)

        # Symmetric proposal -> standard Metropolis (not full Metropolis-Hastings)
        # acceptance ratio, computed in log-space for numerical stability.
        log_accept_ratio = log_post_proposal - log_post_current

        if np.log(rng.uniform()) < log_accept_ratio:
            theta_current = theta_proposal
            log_post_current = log_post_proposal
            n_accepted += 1

        samples[i] = theta_current

    acceptance_rate = n_accepted / n_iter
    return samples, acceptance_rate


if __name__ == "__main__":
    # Minimal smoke test with synthetic data, so this file can be run standalone
    # to confirm the sampler behaves sensibly (posterior mean near the true beta).
    rng = np.random.default_rng(0)
    n = 2000
    true_intercept, true_beta = -2.0, 1.5
    X_demo = rng.binomial(1, 0.3, size=n).astype(float)
    p_demo = 1 / (1 + np.exp(-(true_intercept + true_beta * X_demo)))
    y_demo = rng.binomial(1, p_demo)

    samples, acc_rate = metropolis_hastings(X_demo, y_demo, n_iter=20000, proposal_sd=0.05)
    burn_in = 5000
    post = samples[burn_in:]

    print(f"Acceptance rate: {acc_rate:.2%}")
    print(f"True intercept={true_intercept}, beta={true_beta}")
    print(f"Posterior mean: intercept={post[:,0].mean():.3f}, beta={post[:,1].mean():.3f}")
