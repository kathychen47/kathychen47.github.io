---
title: Alvarez Macovski Method
date: 2023-09-04
mathjax: true
categories:
  - [Computed Tomography, Phyiscs]
tags:
  - attenuation coefficient
disableNunjucks: true
---

## Alvarez Macovski Method

Attenuation coefficient can be represented as a function of $u(E)$

$$
\mu(E)=a_1 f_1(E)+a_2 f_2(E)+\ldots+a_n f_n(E) .
$$

Assume that only photoelectric absorption and Compton effect exist for interaction:

$$
\mu(E)=a_1 \frac{1}{E^3}+a_2 f_{\mathrm{KN}}(E)
$$

- $f_{\mathrm{KN}}(E)$ is the Klein-Nishina function：
  $f_{\mathrm{KN}}(\alpha)=\frac{1+\alpha}{\alpha^2}\left[\frac{2(1+\alpha)}{1+2 \alpha}-\frac{1}{\alpha} \ln (1+2 \alpha)\right]+\frac{1}{2 \alpha} \ln (1+2 \alpha)-\frac{(1+3 \alpha)}{(1+2 \alpha)^2}$

- $\alpha=E / 510.975 \mathrm{keV}$

- $\begin{aligned}
  & a_1 \approx K_1 \frac{\rho}{A} Z^n, \quad n \approx 4 or 3 \\
  & a_2 \approx K_2 \frac{\rho}{A} Z
  \end{aligned}$

  $A$ is atomic weight, $Z$ is atomic number, $\rho$ is mass density

$$ = > \mu (E) = {K*1}\frac{\rho }{A}{Z^3}*\frac{1}{{{E^3}}} + {K_2}\frac{\rho }{A}Z*{f*{{\text{KN}}}}(E)$$

- $E = \frac{{hc}}{\lambda }$

$$ = > \mu (E) = {K*1}\frac{\rho }{A}{Z^3}*\frac{{{\lambda ^3}}}{{{{(hc)}^3}}} + {K_2}\frac{\rho }{A}Z*{f*{{\text{KN}}}}(E)$$

$= > \mu (E) = \frac{{{K_1}}}{{A{{(hc)}^3}}}\rho {Z^3}{\lambda ^3} + \frac{{{K_2}}}{A}\rho Z{f\_{{\text{KN}}}}(E)$$

- ${f_{{\text{KN}}}}(E)$ can be approximate to ${f_{{\text{KN}}}}(E) \propto {E^{ - 1}}$, then ${f_{{\text{KN}}}}(E) \propto \lambda $ ????

Thus:
$$= > \mu (E) = \frac{{{K_1}}}{{A{{(hc)}^3}}}\rho {Z^3}{\lambda ^3} + \frac{{{K_2}}}{A}\rho Z\lambda $$ ????

In tomosipo or gvxr package, attenuation coefficient is calculate by uisng:

$$\mu \left( {material,{\text{ }}E} \right){\text{ }} = {\text{ }}\mu \left( {water,{\text{ }}E} \right){\text{ }} * {\text{ }}\left( {1{\text{ }} + {\text{ }}\frac{{HU\left( {material} \right)}}{{1000}}} \right)$$

${HU\left( {material} \right)}$  and $\mu \left( {water,{\text{ }}E} \right)$ can be found by checking table.

What is the relationship between $\mu \left( {material,{\text{ }}E} \right)$ and $\mu (E)$ we calculated using Alvarez Macovski Method????

${u_\lambda } = {K_1}\rho {Z^3}{\lambda ^3} + {K_2}\rho \lambda$

if we do the projection for object with density $\rho $ and $\rho {Z^3}$, and add them together, it looks like:

$${u_\lambda } = {K_1}{\lambda ^3}(projection{\text{ }}for{\text{ }}object{\text{ }}density{\text{ }}\rho {Z^3}) + {K_2}\lambda (projection{\text{ }}for{\text{ }}object{\text{ }}density{\text{ }}\rho )$$
