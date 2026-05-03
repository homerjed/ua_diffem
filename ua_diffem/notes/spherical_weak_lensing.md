# Spherical weak-lensing simulation note

This note documents the full-sky weak-lensing pipeline used by the spherical
training scripts in `ua_diffem`. It is the physics-focused companion to
[`introduction.md`](introduction.md), which stays centered on the simpler
planar Kaiser-Squires toy problem.

The goal here is not to reproduce every detail of survey weak lensing. The
goal is to make it easy to answer a narrower question:

"What random field is the code sampling, what observation model is it
applying, and what parts of that pipeline are physically motivated versus
deliberately simplified?"

Main code paths:

- [`../shear/glass_maps.py`](../shear/glass_maps.py)
- [`../scripts/train_spherical.py`](../scripts/train_spherical.py)
- [`../scripts/train_spherical_gaussian.py`](../scripts/train_spherical_gaussian.py)

## One-paragraph summary

The spherical code draws a full-sky HEALPix convergence field `kappa` from an
angular power spectrum `C_ell`, optionally using a CAMB spectrum built from a
Planck-2018-plus-BAO `Lambda`CDM cosmology and a Smail source distribution. It
then projects out the unobservable monopole and dipole, converts the remaining
field into an E-mode spin-2 shear map on the sphere, adds independent Gaussian
pixel noise, cuts out random circular survey holes, and passes the normalized
shear-plus-mask channels to the posterior model.

## Forward model in one block

The cleanest code-faithful summary is

$$
\begin{aligned}
\kappa_{\mathrm{latent}} &\sim \text{prior set by } C_\ell, \\
\kappa_{\mathrm{true}} &= P_{\ell \ge 2}\!\left[\kappa_{\mathrm{latent}}\right], \\
\gamma_{\mathrm{true}} &= S\!\left[\kappa_{\mathrm{true}}\right], \\
n &\sim \mathcal{N}\!\left(0, \mathrm{noise\_std}^2 I\right), \\
m &= \text{binary spherical hole mask}, \\
\gamma_{\mathrm{obs}} &= m \, \left(\gamma_{\mathrm{true}} + n\right), \\
\mathrm{condition} &= \left[\gamma_{\mathrm{obs}} / \gamma_{\mathrm{scale}},\, m\right].
\end{aligned}
$$

where:

- `$P_{\ell \ge 2}$` removes the unobservable monopole and dipole;
- `S` is the spherical E-mode convergence-to-shear operator;
- `m` is a 0/1 sky mask;
- `$\gamma_{\mathrm{scale}} = \mathrm{std}(\gamma_{\mathrm{true}}) + \mathrm{noise\_std} + 10^{-6}$`
  in the training script.

The rest of this note expands each term.

## 1. What field is being modeled?

In weak lensing, convergence is a weighted projection of the three-dimensional
matter contrast along the line of sight. Schematically one writes

$$
\kappa(\hat{\mathbf n}) = \int d\chi \, W_\kappa(\chi)\,
\delta\!\left(\chi \hat{\mathbf n}, \chi\right)
$$

with lensing kernel

$$
W_\kappa(\chi) =
\frac{3 H_0^2 \Omega_m}{2 c^2}\,
\frac{\chi}{a(\chi)}\, g(\chi),
$$

with

$$
g(\chi) = \int_\chi^\infty d\chi_s \, n(\chi_s)\,
\frac{\chi_s - \chi}{\chi_s},
$$

and source distribution `n(chi_s)` normalized to unity.

This repository does not simulate `delta(x, z)` directly. Instead it works one
level later: it specifies an angular power spectrum `C_ell^{kappa kappa}` for
`kappa` itself, then samples `kappa` as a random field on the sphere.

That means the code is best viewed as a controlled stochastic benchmark for the
projected lensing field, not as an N-body or ray-tracing pipeline.

## 2. Harmonic representation on the sphere

The spherical code represents convergence as a scalar field on the sky:

$$
\kappa(\hat{\mathbf n}) =
\sum_{\ell,m} \kappa_{\ell m} Y_{\ell m}(\hat{\mathbf n}).
$$

For an isotropic Gaussian random field, the harmonic coefficients obey

$$
\left\langle
\kappa_{\ell m}\, \kappa_{\ell' m'}^\ast
\right\rangle
=
C_\ell^{\kappa\kappa}\,
\delta_{\ell\ell'} \delta_{mm'}.
$$

This covariance statement is the prior that the code implements when it calls
GLASS to draw Gaussian `kappa` maps.

The HEALPix discretization used by the code is:

$$
N_{\mathrm{pix}} = 12\, n_{\mathrm{side}}^2,
\qquad
\ell_{\max}^{\mathrm{default}} = 3\, n_{\mathrm{side}} - 1.
$$

Relevant code:

- [`../shear/glass_maps.py`](../shear/glass_maps.py):
  `default_lmax`, `generate_glass_gaussian_kappa_dataset`,
  `generate_glass_lognormal_kappa_dataset`

## 3. How the code chooses `C_ell`

Two families are implemented.

### 3a. Toy spectrum

The simplest option is a hand-built decaying spectrum:

$$
C_\ell =
A\, (\ell + 1)^{-\mathrm{spectral\_index}}
\exp\!\left[-\mathrm{damping}\,\ell(\ell + 1)\right].
$$

implemented by `glass_kappa_cls`.

This is not a cosmological first-principles model. It is a convenient smooth
prior with tunable large-scale power and high-`ell` damping.

### 3b. CAMB-backed spectrum

The more physical option uses CAMB via `camb_lensing_kappa_cls`.
Conceptually, this is the route

$$
\text{cosmology} + \text{source distribution}
\;\to\;
\text{lensing kernel}
\;\to\;
C_\ell^{\kappa\kappa}.
$$

The code does not evaluate the line-of-sight projection formula by hand.
Instead it constructs a source window and asks CAMB for the corresponding
source angular spectrum.

The source redshift distribution is the Smail form

$$
\frac{dN}{dz} \propto z^\alpha
\exp\!\left[-\left(\frac{z}{z_0}\right)^\beta\right].
$$

normalized over `0 <= z <= zmax`.

This is a generic single-bin forecast-style choice. It is not a survey
selection function fitted to a specific real catalog.

Relevant code:

- [`../shear/glass_maps.py`](../shear/glass_maps.py):
  `smail_redshift_distribution`, `camb_lensing_kappa_cls`, `make_kappa_cls`

## 4. Default cosmological parameters and where they come from

The default CAMB parameters in the spherical scripts are

$$
\begin{aligned}
H_0 &= 67.66, \\
\Omega_b h^2 &= 0.02242, \\
\Omega_c h^2 &= 0.11933, \\
n_s &= 0.9665, \\
A_s &= 2.105 \times 10^{-9}, \\
\sum m_\nu &= 0.06\,\mathrm{eV}.
\end{aligned}
$$

These match the `Planck TT,TE,EE+lowE+lensing+BAO` base-`Lambda`CDM values from
the final Planck 2018 release. Equivalently,

$$
\ln\!\left(10^{10} A_s\right) = 3.047,
$$

corresponds to `As = 2.105e-9`.

Primary references:

- Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters,"
  *Astronomy & Astrophysics* **641** (2020), A6.
- Planck Collaboration, "Planck 2018 results. I. Overview, and the
  cosmological legacy of Planck," *Astronomy & Astrophysics* **641** (2020),
  A1, especially the summary table.

So the default spherical setup is using a Planck-2018-plus-BAO flat
`Lambda`CDM background, plus a separate generic Smail source distribution.

## 5. Gaussian versus log-normal latent fields

The code can generate either a Gaussian or a skewed log-normal-like latent
field.

### Gaussian option

`generate_glass_gaussian_kappa_dataset` samples a Gaussian HEALPix map from the
input `C_ell`.

If `gaussian_remove_dipole` is enabled, the code removes the dipole before the
observable-mode projection. Because this step is linear, the field remains
Gaussian in the surviving harmonic subspace.

### Log-normal option

`generate_glass_lognormal_kappa_dataset` starts from a Gaussian field `g`,
normalizes it to mean zero and variance one, then applies

$$
\kappa_{\mathrm{LN}} =
\exp\!\left(\sigma g - \frac{1}{2}\sigma^2\right) - 1.
$$

The subtraction of `0.5 * sigma^2` makes the transformed field have mean near
zero before the later harmonic projection.

This log-normal step is a phenomenological choice. It is meant to inject
positive skewness and some visual non-Gaussian character associated with matter
clustering, but it is not a full nonlinear structure-formation model.

One important implementation detail:

1. the nonlinear log-normal transform happens first;
2. the code then projects out low-`ell` modes with `ell < 2`.

So the final latent target is not exactly a textbook log-normal field. It is a
log-normal-like field projected into the shear-observable harmonic subspace.

## 6. Why the code removes `ell = 0, 1`

The spherical trainer explicitly projects the convergence field into the
observable subspace:

$$
\kappa_{\mathrm{true}} =
P_{\ell \ge 2}\!\left[\kappa_{\mathrm{latent}}\right].
$$

implemented by `project_observable_kappa_numpy`.

In harmonic language this means

$$
\kappa_{\ell m} \to 0
\qquad \text{for } \ell = 0, 1,
$$

before transforming back to map space.

Physics meaning:

- `ell = 0` is the monopole, a sky-wide constant offset;
- `ell = 1` is the dipole;
- shear does not constrain those modes in the spherical inversion used here.

This projection matters because otherwise the model would be trained to predict
large-scale convergence components that the observation operator can never
recover from shear.

Relevant code:

- [`../shear/glass_maps.py`](../shear/glass_maps.py):
  `project_observable_kappa_numpy`, `_observable_alm_keep_mask`

## 7. From spherical convergence to spherical shear

The code converts `kappa` to shear with `spherical_shear_numpy`, which calls
`glass.from_convergence(..., shear=True)`.

The physically important distinction from the planar note is that shear on the
sphere is a spin-2 field. A convenient decomposition is

$$
\gamma(\hat{\mathbf n}) =
\sum_{\ell,m}
\left(\gamma^E_{\ell m} + i \gamma^B_{\ell m}\right)\,
{}_2Y_{\ell m}(\hat{\mathbf n}),
$$

where `_2Y_(ell m)` are spin-2 spherical harmonics.

Because the simulator starts from a scalar convergence field, the noiseless
truth shear is pure E-mode:

$$
\gamma^B_{\ell m} = 0.
$$

For the convention used by the code, the spherical harmonic relation between
convergence and E-mode shear is

$$
\gamma^E_{\ell m}
=
-\sqrt{\frac{(\ell + 2)(\ell - 1)}{\ell(\ell + 1)}}\,
\kappa_{\ell m},
$$

for `ell >= 2`. Equivalently,

$$
\kappa_{\ell m}
=
-\sqrt{\frac{\ell(\ell + 1)}{(\ell - 1)(\ell + 2)}}\,
\gamma^E_{\ell m},
$$

This inverse factor is exactly the response encoded in `_ks_response` and later
used by `spherical_kaiser_squires_numpy`, up to the pixel-window correction
described below.

So, from a physics perspective, the spherical forward operator is:

$$
\text{scalar } \kappa \text{ field}
\;\to\;
\text{pure-E spin-2 shear field}.
$$

and not a general `E+B` spin-2 simulator.

## 8. Pixel ordering and pixel windows

Two implementation details are easy to misread as physics if they are not
called out explicitly.

### NESTED versus RING ordering

The training code stores maps in NESTED HEALPix ordering, but some spherical
harmonic operations are easier to call through `healpy` in RING ordering.
Functions such as `reorder_healpix_array` switch between those layouts.

This changes storage order only. It does not change the field itself.

### Pixel-window correction

The spherical inverse uses a response factor multiplied by a ratio of HEALPix
pixel windows:

$$
\mathrm{response}_\ell
\leftarrow
\mathrm{response}_\ell\,
\frac{pw_0(\ell)}{pw_2(\ell)}.
$$

This compensates for the different smoothing associated with scalar and spin-2
HEALPix sampling. It is a discretization correction, not a new physical term
in the lensing equations.

Relevant code:

- [`../shear/glass_maps.py`](../shear/glass_maps.py):
  `_ks_response`, `reorder_healpix_array`

## 9. Observation model: noise, mask, and normalized condition

After generating `gamma_true`, the code applies the corruption model

$$
\gamma_{\mathrm{obs}} =
m\,\left(\gamma_{\mathrm{true}} + n\right).
$$

with:

$$
\left\langle n_a(p)\, n_b(p') \right\rangle =
\mathrm{noise\_std}^2 \,
\delta_{ab}\, \delta_{pp'}.
$$

where:

- `a, b` label the two shear components;
- `p, p'` label pixels;
- `m(p)` is either 0 or 1.

In words, the noise is:

- additive;
- Gaussian;
- independent between pixels;
- independent between `gamma1` and `gamma2`;
- spatially uniform.

That makes it a simplified stand-in for shape noise, not a galaxy-sampled noise
field with varying depth or weights.

The model condition is

$$
\mathrm{condition} =
\begin{bmatrix}
\gamma_{\mathrm{obs},1} / \gamma_{\mathrm{scale}} \\
\gamma_{\mathrm{obs},2} / \gamma_{\mathrm{scale}} \\
m
\end{bmatrix},
$$

with

$$
\gamma_{\mathrm{scale}} =
\mathrm{std}\!\left(\gamma_{\mathrm{true}}\right) +
\mathrm{noise\_std} + 10^{-6},
$$

from [`../scripts/train_spherical.py`](../scripts/train_spherical.py).

This scaling is purely numerical. It is not a physical parameter of the
cosmology.

Relevant code:

- [`../shear/glass_maps.py`](../shear/glass_maps.py):
  `spherical_corruption_from_shear_numpy`,
  `SphericalShearObservationChannel`
- [`../scripts/train_spherical.py`](../scripts/train_spherical.py):
  construction of `gamma_scale`

## 10. How the spherical mask is generated

The mask model is a random union of circular holes on the sphere.

Each realization:

1. starts from full sky;
2. samples `num_holes` random unit vectors as candidate hole centers;
3. draws whether each candidate hole is active with probability
   `mask_fraction`;
4. removes pixels within angular radius `hole_radius_deg` of every active
   center.

So `mask_fraction` is not the final masked area fraction. It is the activation
probability per candidate hole. The actual missing sky area depends on:

- `num_holes`;
- `hole_radius_deg`;
- overlap between holes.

In equations, the code is closer to

$$
m(\hat{\mathbf n}) = \prod_i \left[1 - h_i(\hat{\mathbf n})\right],
$$

where each `h_i` is either a circular binary hole or identically zero.

This is a deliberately simple survey-footprint model. It does not simulate
depth variations, complex boundaries, star masks, or inhomogeneous weighting.

Relevant code:

- [`../shear/glass_maps.py`](../shear/glass_maps.py):
  `generate_spherical_hole_masks`

## 11. Bootstrap reconstruction and what it means physically

For diagnostics and the first EM iteration, the code computes a deterministic
baseline reconstruction from observed shear:

$$
\gamma_{\mathrm{obs}} \to \kappa_{\mathrm{KS}}.
$$

using `spherical_kaiser_squires_numpy`.

Ignoring masking for a moment, the harmonic inverse is

$$
\kappa_{\ell m}
=
-\sqrt{\frac{\ell(\ell + 1)}{(\ell - 1)(\ell + 2)}}\,
\gamma^E_{\ell m},
$$

followed by the same `ell >= 2` projection used in the latent-target
construction.

With a mask, the code does not treat masked pixels as genuinely observed zero
shear. Instead it iterates:

1. invert the current filled shear map;
2. forward-project that `kappa` back to shear;
3. replace only the masked pixels with the forward prediction;
4. keep observed pixels fixed.

This is better interpreted as a simple masked harmonic inpainting bootstrap
than as a posterior mean.

Relevant code:

- [`../shear/glass_maps.py`](../shear/glass_maps.py):
  `spherical_kaiser_squires_numpy`
- top-level note on the earlier masking bug:
  [`../../notes/issues.md`](../../notes/issues.md)

## 12. What is physically realistic, and what is idealized?

The spherical pipeline has several physically meaningful ingredients:

- full-sky spherical geometry through HEALPix;
- a spin-2 shear field generated from a scalar convergence field;
- an optional CAMB-backed convergence spectrum;
- an explicit source distribution for the lensing kernel;
- removal of convergence modes that are unobservable from shear;
- explicit masking and additive shear noise.

It also makes strong simplifications:

- it uses shear `gamma`, not reduced shear
  `$g = \gamma / (1 - \kappa)$`;
- it simulates only E-mode truth shear from scalar convergence;
- it does not inject intrinsic alignments, PSF systematics, multiplicative
  shear bias, or photo-z errors;
- it uses independent pixel noise instead of a galaxy catalog with weights;
- it uses a random-hole mask instead of a survey-specific footprint;
- the log-normal prior is phenomenological rather than a full nonlinear matter
  simulation;
- there is no tomography beyond the single effective source window used to
  construct `C_ell`.

So the right mental model is:

*physics-informed synthetic benchmark*

not

*end-to-end survey realism*.

## 13. Audit checklist for a saved run

If you want to review a run such as
[`../../runs/ua_diffem/spherical_basic_2/config.json`](../../runs/ua_diffem/spherical_basic_2/config.json)
from a physics perspective, these are the highest-value fields to inspect
first:

- `kappa_power_spectrum`: `toy` or `camb`;
- `camb.*`: the background cosmology used to compute the source spectrum;
- `source_distribution.*`: the effective source-population model;
- `prior_family`: `gaussian` or `lognormal`;
- `gaussian_sigma`: strength of the log-normal skew transform;
- `nside` and `lmax`: angular resolution and harmonic cutoff;
- `observable_ell_min`: the low-mode cutoff, fixed to `2` in the current code;
- `noise_std`: additive shear-noise level;
- `num_holes`, `hole_radius_deg`, `mask_fraction`: mask geometry;
- `gamma_scale`: normalization only, not a physical observable.

## 14. Minimal code-to-physics map

If you are reading the code itself, this is the fastest route:

1. [`../shear/glass_maps.py`](../shear/glass_maps.py):
   `make_kappa_cls` for the power spectrum choice.
2. [`../shear/glass_maps.py`](../shear/glass_maps.py):
   `generate_glass_gaussian_kappa_dataset` and
   `generate_glass_lognormal_kappa_dataset` for latent `kappa`.
3. [`../shear/glass_maps.py`](../shear/glass_maps.py):
   `project_observable_kappa_numpy` for the `ell < 2` projection.
4. [`../shear/glass_maps.py`](../shear/glass_maps.py):
   `spherical_shear_numpy` for `kappa -> gamma`.
5. [`../shear/glass_maps.py`](../shear/glass_maps.py):
   `spherical_corruption_from_shear_numpy` for noise and masking.
6. [`../shear/glass_maps.py`](../shear/glass_maps.py):
   `spherical_kaiser_squires_numpy` for the baseline inverse.
7. [`../scripts/train_spherical.py`](../scripts/train_spherical.py):
   assembly of the dataset, normalization, and saved config.
