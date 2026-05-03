# Issues

## Masked Spherical KS Bootstrap

### Problem

The spherical bootstrap path used to pass a mask into the Kaiser-Squires
inverse, but the inverse treated masked pixels as literal zero shear rather
than as missing data. In practice that can suppress reconstructed kappa
amplitudes and seed the first EM round with overly conservative latent maps.

### Stage-1 Fix

The current stage-1 fix is an iterative mask-aware bootstrap:

1. Start from the observed shear map, where masked pixels are zeroed.
2. Run the spherical Kaiser-Squires inverse to get a kappa estimate.
3. Forward-project that kappa estimate back into shear space.
4. Keep the observed shear fixed on unmasked pixels.
5. Refill only the masked pixels with the forward-projected shear.
6. Repeat a few iterations, then do one final KS inverse.

This is still an approximation, but it avoids the specific bug where missing
shear was implicitly interpreted as measured zero shear.

### Follow-Up Options

- Apodize the binary mask before harmonic transforms to reduce ringing.
- Replace the bootstrap with a Wiener / MAP solve for the Gaussian prior case,
  then draw constrained realizations from the Gaussian posterior.
- Add a supervised warm-start phase for the conditional posterior model before
  running EM, so the method does not rely on a weak analytic bootstrap.

### Useful Controls

- No-mask control: set `mask_fraction=0` or `num_holes=0`.
- Lower-noise control: start with `noise_std=0.2` and compare against `0.1`
  and `0.3`.
