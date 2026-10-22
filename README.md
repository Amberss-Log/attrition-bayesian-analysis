# Bayesian Attrition Analysis — People Analytics Case Study

A statistical case study that applies inferential statistics and Bayesian methods
(from my coursework in Probability & Statistics and Bayesian Statistical Methods)
to a public workforce dataset. The same toolkit applies to customer churn: what
drives people to leave, and how confident can we be in each driver?

## What this is

Given a public HR dataset, the goal was to answer a concrete business
question — **what actually drives employee attrition, and how confident can
we be in each factor?** The approach:
- hypothesis testing to screen candidate drivers,
- a Bayesian model to quantify each effect with proper uncertainty, and
- a plain-language summary that a non-technical stakeholder could act on.

## Key findings

| Driver | Effect (adjusted odds ratio, 94% credible interval) |
|---|---|
| Overtime (yes vs no) | **4.5x** the odds of leaving (3.3–6.1) |
| Single (vs divorced) | 3.3x (2.2–5.0) |
| Total working experience, per +7.8 years (1 SD) | 0.53x, so the odds roughly halve (0.44–0.63) |
| Distance from home, per +8.1 km (1 SD) | 1.3x (1.1–1.5) |
| Job satisfaction, per +1.1 points (1 SD) | 0.69x (0.60–0.80) |
| Married (vs divorced) | ambiguous (0.85–2.0) |

- **Odds are not rates:** overtime workers actually left at **30.5% vs 10.4%**, a **2.9x** rate ratio. The 4.5x figure is an *odds* ratio, adjusted for the other drivers.
- **Highest-risk segment:** early-career employees (≤ 5 years of total experience) who work overtime. **55.6%** of them left (n = 90), against 8.3% of experienced employees without overtime.
- **Is overtime worse for early-career staff? (interaction model)**
  - **Probability scale:** clearly yes. Overtime adds **~35 percentage points** of attrition for early-career staff vs **~16** for experienced staff.
  - **Odds scale:** possibly. The odds ratio is 5.9x vs 3.8x, with an 89% probability that the difference is real, but the interval includes no difference. The segment's high rate is mostly two risk factors stacking.
- **Impact estimate:** without overtime, the model expects about **32 of that segment's 50 departures to be avoided** (94% CrI 22–41). This group is 22% of overtime workers but ~39% of the avoidable departures. This is a planning estimate that assumes the association is causal.

Full write-up: [`outputs/executive_summary.md`](outputs/executive_summary.md) ([PDF](outputs/executive_summary.pdf)).

## Repo structure

```
attrition-bayesian-analysis/
├── data/                            # IBM HR dataset + cleaned copy (not versioned, see .gitignore)
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory analysis, cleaning
│   ├── 02_hypothesis_testing.ipynb  # Chi-square, Welch's t-test, Mann-Whitney U + effect sizes
│   └── 03_bayesian_model.ipynb      # Bayesian logistic regression (PyMC) + from-scratch MH validation
├── src/
│   └── metropolis_hastings.py       # Standalone random-walk MH sampler (numpy only)
├── outputs/
│   ├── hypothesis_test_results.csv
│   ├── bayesian_model_convergence_summary.csv
│   ├── bayesian_model_odds_ratios.csv
│   ├── segment_attrition_rates.csv
│   ├── impact_estimate.csv
│   ├── forest_plot.png
│   ├── executive_summary.md
│   └── executive_summary.pdf
└── README.md
```

## Method

1. **EDA:** cleaned the dataset (1,470 employees, 35 columns). Checked for nulls, duplicates and constant columns, and dropped 4 uninformative columns.
2. **Hypothesis testing:** screened candidate drivers with chi-square tests (categorical), Welch's t-tests (numeric) and Mann-Whitney U tests (ordinal 1–4 scales). Each test reports an effect size: Cramér's V, Cohen's d, or rank-biserial correlation. At n = 1,470 almost everything is "significant", so effect size is what separates strong drivers from weak ones.
3. **Feature selection:** income, age and total working years are strongly correlated (r = 0.50–0.77). Only total working years enters the model, which avoids unstable coefficients from multicollinearity.
4. **Bayesian logistic regression:** fitted in PyMC with weakly informative Normal(0, 2) priors and sampled with NUTS (4 chains). Convergence is good (r̂ = 1.00, bulk ESS > 4,000).
5. **Sampler validation:** a from-scratch random-walk Metropolis-Hastings sampler (`src/metropolis_hastings.py`) and PyMC/NUTS fit **the same** overtime-only model, and their posteriors match (β = 1.314 vs 1.318, sd 0.145 vs 0.146). The step size was tuned from 0.05 (73% acceptance, ESS ≈ 100) to 0.25 (21% acceptance, ESS ≈ 2,400).
6. **Model checks:** posterior predictive simulation reproduces the attrition rate overall (16.2% vs 16.1%) and within each overtime group.
7. **Interaction model and what-if:** a second model adds an overtime × early-career interaction to test the segment finding formally, on both the odds and the probability scale. A counterfactual (overtime on vs off for the same employees) sizes the impact. LOO cross-validation shows both models predict about equally well.
8. **Communication:** converted log-odds into odds ratios with credible intervals, and wrote a one-page, jargon-free executive summary.

## How this transfers to product analytics

The same workflow applies to **user churn** in a consumer app such as a payments app:

| This project | Payments-app equivalent |
|---|---|
| Employee leaves | User stops transacting for 30+ days |
| Overtime, experience, distance | Failed transactions, days since sign-up, cashback received |
| Early-career × overtime segment | New users who hit a payment failure in their first week |
| "Departures avoided" what-if | "Churned users avoided if first-week failures were halved", to size a fix before building it |
| Overtime pilot | A/B test, e.g. instant auto-retry on failed payments for new users |

## Data

[IBM HR Analytics Employee Attrition & Performance dataset](https://www.kaggle.com/datasets/pavansubhash/ibm-hr-analytics-attrition-dataset)
(public, Kaggle). This is synthetic data released by IBM for analytics practice, not real company data.

## Caveats

- This is an **association** study, not a causal one.
- The data is a single cross-sectional snapshot.
- The model has 6 predictors for 237 attrition events, which is adequate but not large.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda.ipynb   # then 02, then 03
```
