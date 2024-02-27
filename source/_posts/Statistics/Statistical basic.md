---
title: Some basic things in Statistics
date: 2024-02-01 00:00:00
mathjax: true
categories:
	- [Statistics]
---

# Some basic things in Statistics

## Mean, Variance, Standard deviation, Covariance, Correlation

- mean of a vector $\underline x$ is $\mathrm{mean}(\underline x) = \frac{1}{n} \sum_{i=1}^n x_i$.
- variance of a vector $\underline x$ is $\mathrm{var}(\underline x) =  \frac{1}{n-1} \sum_{i=1}^n (x_i-\mathrm{mean}(x))^2$.
- standard deviation of a vector $\underline x$ is $\mathrm{sd}(\underline x) = \sqrt{\mathrm{var}(\underline x)}$.
- covariance of two vectors $\underline x,\underline y$ of _same length_ is $\mathrm{cov}(\underline x,\underline y) = \frac{1}{n-1}\sum_{i=1}(x_i-\mathrm{mean}(\underline x))\cdot (y_i-\mathrm{mean}(\underline y))$.
- correlation of two vectors $\underline x,\underline y$ of _same length_ is $\mathrm{corr}(\underline x,\underline y) = \frac{\mathrm{cov}(x,y)}{\mathrm{sd}(\underline x) \cdot \mathrm{sd}(\underline y) }$ and is between $-1$ and $1$

**Some tips:**

1. Covariance and correlation are both measures of the relationship between two variables, but they differ in several key ways:

   - Covariance is unnormalized and challenging to compare across variables with different units.
   - Correlation is normalized between -1 and +1, indicating both the direction and strength of the linear relationship.

2. Correlation measures only linear relationships and does not imply causation.

## t-distribution

The t-distribution is a probability distribution typically used to estimate the `mean` of a normally distributed population when the `sample size is small` (such as less than 30) and the population `standard deviation is unknown`.

> As sample sizes increase, the t-distribution's shape gradually approaches that of the normal distribution due to the effect of the **Central Limit Theorem**.

## Gaussian distribution (normal distribution)

| Feature                 | t-Test                                                                                                                                          | Z-test                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Applicability**       | Unknown population standard deviation, small sample size (usually n<30)                                                                         | Known population standard deviation or large sample size (n≥30)                                                             |
| **Tail Thickness**      | Thicker tails, providing wider confidence intervals                                                                                             | Thinner tails, resulting in narrower confidence intervals                                                                   |
| **Degrees of Freedom**  | Yes (sample size minus 1)                                                                                                                       | No                                                                                                                          |
| **Critical Value**      | t-score, depends on sample's degrees of freedom                                                                                                 | z-score, based on the standard normal distribution                                                                          |
| **Coverage**            | Generally wider, more conservative                                                                                                              | More precise, less wide                                                                                                     |
| **Usage**               | Confidence intervals, hypothesis testing (small samples)                                                                                        | Confidence intervals, hypothesis testing (large samples or known σ)                                                         |
| **t-score & z-score**   | t-score: $t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$ <br />where $\bar{x}$ is sample mean, $s$ is sample std. dev.                                 | z-score: $z = \frac{\bar{x} - \mu}{\sigma / \sqrt{n}}$ <br>where $\bar{x}$ is sample mean, $\sigma$ is population std. dev. |
| **Confidence Interval** | $\bar{x} \pm t_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$ <br>where $t_{\alpha/2}$ is the t critical value for $\alpha/2$ and $n-1$ degree of freedom | $\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$ <br>where $z_{\alpha/2}$ is the z critical value for $\alpha/2$    |

**Some Tips**:

1. The Gaussian distribution does not involve degrees of freedom because it is not designed to address uncertainties arising from sample size. Instead, it is a universal model applicable for describing overall data distributions, with its shape completely determined by the mean and standard deviation.
