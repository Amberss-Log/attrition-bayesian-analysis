# Bayesian Attrition Analysis — People Analytics Case Study

A statistical case study applying inferential statistics and Bayesian methods
(from coursework and an ongoing BTP in Hamiltonian Monte Carlo) to a real
workforce dataset, to practice the kind of analysis used in people-analytics
and applied-research roles.

## What this is

Given a public HR dataset, the goal was to answer a concrete business
question — **what actually drives employee attrition, and how confident can
we be in each factor?** — using the same toolkit an applied research role
would: hypothesis testing to screen candidate drivers, a Bayesian model to
quantify effects with proper uncertainty, and a plain-language summary a
non-technical stakeholder could act on.

## Key finding

Working overtime is associated with **~4.5x higher odds** of attrition — by
far the strongest driver identified — and tenure is strongly protective
(each step up in experience roughly halves the odds). Full write-up in
[`outputs/executive_summary.pdf`](outputs/executive_summary.pdf).

## Repo structure

```
attrition-bayesian-analysis/
├── data/                          # WA_Fn-UseC_-HR-Employee-Attrition.csv (not versioned, see .gitignore)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory analysis, cleaning
│   ├── 02_hypothesis_testing.ipynb  # Chi-square, Welch's t-test, Mann-Whitney U + effect sizes
│   └── 03_bayesian_model.ipynb      # Bayesian logistic regression (PyMC) + from-scratch MH validation
├── src/
│   └── metropolis_hastings.py       # Standalone random-walk MH sampler, used to validate the PyMC posterior
├── outputs/
│   ├── hypothesis_test_results.csv
│   ├── bayesian_model_odds_ratios.csv
│   ├── forest_plot.png
│   ├── executive_summary.md
│   └── executive_summary.pdf
└── README.md
```

## Method

1. **EDA** — cleaned the dataset (1,470 employees, 35 features), checked for
   nulls/duplicates/constant columns, and visualized attrition patterns
   across categorical, numeric, and ordinal features.
2. **Hypothesis testing** — screened candidate drivers with chi-square tests
   (categorical), Welch's t-tests (numeric), and Mann-Whitney U tests
   (ordinal), reporting effect sizes (Cramér's V, Cohen's d) alongside
   p-values, since statistical significance and practical significance
   aren't the same thing at n=1,470.
3. **Feature selection** — checked pairwise correlations to avoid feeding
   collinear features (e.g. tenure, income, and age all move together) into
   the model, which would make individual coefficients unstable.
4. **Bayesian logistic regression** — fit with PyMC using weakly informative
   priors, sampled with NUTS, and checked convergence (`r_hat`, effective
   sample size) before trusting any result.
5. **Validation** — implemented a random-walk Metropolis-Hastings sampler
   from scratch (`src/metropolis_hastings.py`) for a simplified
   single-predictor version of the model, and compared its posterior against
   PyMC's, as a check on the sampling mechanics themselves.
6. **Communication** — converted posterior log-odds into odds ratios with
   credible intervals, and wrote a one-page, jargon-free summary for a
   non-technical audience.

## Data

[IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhash/ibm-hr-analytics-attrition-dataset)
(public, Kaggle). Synthetic/anonymized data released by IBM for analytics
practice — not real company data.

## Caveats

This is an association study, not a causal one — the model identifies which
factors move together with attrition, not proof that any one of them causes
it. See the [executive summary](outputs/executive_summary.md) for the full
caveat and recommendation.

## Running it

```bash
pip install pandas numpy scipy pymc arviz matplotlib seaborn
jupyter notebook notebooks/01_eda.ipynb
```
