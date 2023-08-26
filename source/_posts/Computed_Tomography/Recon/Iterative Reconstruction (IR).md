---
title: Iterative reconstruction (IR)
date: 2023-04-07
mathjax: true
categories:
  - [Computed Tomography, Reconstruction]
tags:
  - IR
disableNunjucks: true
---

Algebraic reconstruction technique includes:

1. Algebraic Iterative Reconstruction

   - Algebraic Reconstruction technique (ART)
   - Multiplicative Algebraic Reconstruction technique (MART)
   - Simultaneous Algebraic Reconstruction Technique (SART)

2. Statistical Iterative Reconstruction

   - Maximum Likelihood Expectation Maximum (MLEM)

   - Ordered Subsets Expectation Maximization (OSEM)

## Algebraic Reconstruction Techniques (ART)

The problem of Iterative reconstruction problem can be reduced to solving the following linear equation:

$$
\left\{ \matrix{
  {\omega _{11}}{f_1} + {\omega _{12}}{f_2} +  \cdots  + {\omega _{1n}}{f_n} = {p_1} \hfill \cr
  {\omega _{21}}{f_1} + {\omega _{21}}{f_2} +  \cdots  + {\omega _{2n}}{f_n} = {p_2} \hfill \cr
  {\rm{                      }} \vdots  \hfill \cr
  {\omega _{m1}}{f_1} + {\omega _{m1}}{f_2} +  \cdots  + {\omega _{mn}}{f_n} = {p_m} \hfill \cr}  \right.
$$

$omega$ : projection data

$p$ : projection matrix (sinogram)

$f$ : Pixel value

$m$: the number of projected angles, $i=1, 2, ..., m$

$n$ : the number of pixel value in the whole image, $j=1, 2, ..., n$

It can be wrote as:

$$
wf=p
$$

**Iterative formula for ART:**

$$
f_j^{(k + 1)} = f_j^{(k)} + \lambda {{{p_i} - \sum\limits_{i = 1}^N {{\omega _{in}}f_n^{(k)}} } \over {\sum\limits_{i = 1}^N {\omega _{in}^2} }}{\omega _{ij}}
$$

$k$ : the number of iterations

$i$ : projected angle, $i=1, 2, ..., m$

$j$ : $j$ th pixel values of the whole $n$ pixel values , $j=1, 2, ..., n$

$p_i$ : measured projection data (real sinogram)
