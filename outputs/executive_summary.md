# Employee Attrition: Key Drivers
*Executive Summary*

Based on a statistical analysis of 1,470 employee records. It combines hypothesis testing with Bayesian modelling to find which factors are most reliably linked to attrition, and how strong each effect is.

## What we found

- **Overtime is the single biggest signal.** 30.5% of employees who work overtime left, compared with 10.4% of those who don't. That is about **3x the rate**. After accounting for the other factors, overtime is linked to about 4.5x the *odds* of leaving.
- **Experience protects against attrition.** Employees with more total working experience are much less likely to leave. Every ~8 additional years of experience roughly halves the odds.
- **The highest-risk group is early-career employees on overtime.** Among employees with 5 years of experience or less who work overtime, **more than half (56%) left**. Among experienced employees without overtime, only 8% did.
- **Single employees leave more often** than divorced employees, at about 3.3x the odds. Married and divorced employees show no clearly different risk.
- **Two smaller but real effects:** employees who live farther from work, and employees who report lower job satisfaction, show moderately higher attrition risk. Both effects are real but much smaller than overtime or experience.

## What this means

Overtime and experience tell one coherent story: **newer employees under heavy workload are the group most likely to leave.** If retention efforts need to be prioritised, this segment is the place to start.

## A caveat worth stating plainly

This analysis shows association, not proof of cause. The data alone can't tell us whether overtime causes people to leave, or whether people already planning to leave end up with different workloads. The findings show where to look and what to prioritise. They are not a guaranteed fix.

## Recommendation

Run a focused **pilot for early-career employees (≤ 5 years of experience) who work overtime**, for example capping overtime or adding proactive manager check-ins. Compare their attrition against a similar group without the change. This tests the effect directly, instead of inferring it from historical data.

**Why this group:** they are only about a fifth of overtime workers, but they account for roughly **40% of the departures that removing overtime could prevent**. If the pattern is causal, the model estimates that about **32 of their 50 departures** could have been avoided (plausible range 22–41).

---

*Methodology: chi-square, Welch's t-test and Mann-Whitney U tests with effect sizes screened candidate drivers. A Bayesian logistic regression (PyMC/NUTS) estimated each driver's effect as an odds ratio with a 94% credible interval. A hand-written Metropolis-Hastings sampler reproduced the PyMC posterior on a simplified model as a check. An interaction model and an overtime on/off what-if sized the impact for the highest-risk segment. Data: IBM HR Analytics dataset (public, synthetic).*
