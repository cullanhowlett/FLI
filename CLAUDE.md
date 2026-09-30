# FLI — project notes

Fork of [hsimonfroy/benchmark-field-level](https://github.com/hsimonfroy/benchmark-field-level) (`upstream` remote),
being substantially modified. Package: `flbench`. Origin: `git@github.com:cullanhowlett/FLI.git`.

## Goal
Field-level reconstruction of the initial (linear) density field from final (z=0) tracer positions and velocities.
Test on synthetic N-body truths first, then on AbacusSummit: z=0 halos (pos, vel) vs. the Abacus initial density
fields (or the z~8 snapshot of halos / 10% particle subsets as a fallback reference).

## Workflow
- Claude edits files directly in the local repo; Cullan runs code on his laptop (PyCharm, conda env `FLI`, CPU JAX).
- Heavy runs (large meshes, sampling) go on the HPC, without an agent.
- PyCharm Community opens notebooks read-only, so work in `.py` scripts; the notebooks in `examples/` are upstream reference.
- Code style: concise, no unnecessary docstrings/scaffolding, lines up to ~130 chars, helpers/comprehensions over repetition.

## Environment
- `pip install -e .` in a Python 3.10–3.12 env; `brew install graphviz` for `model.render()`.
- jax_cosmo imports `pkg_resources`: needs `pip install "setuptools<81"`.
- Code imports blackjax internals (`progress_bar.gen_scan_fn`, `mcmc.adjusted_mclmc_dynamic`); pin `blackjax==1.2.5` if they break.
- Newer numpyro (>=0.19) rejects `DetruncTruncNorm` (`high=inf`); numpyro 0.18 / jax 0.6 known to work.

## Scripts
- `examples/make_truth.py`: N-body truth at a=1 (`evolution='nbody'`, BullFrog). Saves to `examples/save_dir/` (git-ignored):
  `truth.npz` with `init_field` (linear delta, real space, normalised to a=1; scale by D(z)/D(0) for other epochs),
  `final_field` (z=0 real-space 1+delta, CIC from all particles), `pos` [Mpc/h] and `vel` [km/s] for a random `frac` subset,
  true params, `seed`, `frac`, `config` (str(model)); plus `model.yaml`. If a saved truth matches the current config/params/seed/frac
  it is reloaded and only the plots are remade (`remake=True` forces a rerun). Plots: thin-slab slices (`slab`, `axis`) with
  mass-weighted velocity arrows from the saved subset, shot-noise-subtracted P(k) vs linear/halofit, velocity histograms.

## Changes to upstream physics (and why)
- Particle lattice starts at cell centres (`+0.5` in `FieldLevelModel.evolve`). On CIC nodes the kernel has a kink, giving a
  half-cell phase error that decorrelated grid-scale modes and worsened with more steps.
- `pm_forces`: painted fields are deconvolved by the paint+read CIC window (sinc^4). Fields read at the Lagrangian lattice
  (initial velocities, LPT, other solvers) are deconvolved by the exact cell-centre window `lattice_read_kernel`
  (prod cos(k_i/2); Nyquist planes zeroed). Same correction in `lagrangian_weights` (exact for b1 only).
- N-body velocities recorded as deterministic site `nbody_vel` (growth-time units dq/dg, mesh units;
  physical v = a H(a) f(a) g(a) vel * cell_size).
- Validation: 512 Mpc/h, 128^3, 5 steps agrees with halofit to ~10% up to k~0.7 h/Mpc after shot-noise subtraction.

## Open items
- Fit script: load `truth.npz` + `model.yaml`, MCLMC warmup + sampling (from `infer_model.ipynb`); compare reconstructed
  vs. true initial field with `metrics.powtranscoh` (r(k), T(k)). Small mesh locally first.
- Separate particle lattice from force/observation mesh (`ptcl_shape` vs `mesh_shape`): new config, lattice offset on the
  finer mesh, paint normalisation, generalised read kernel, IC padding. Proper fix for grid-scale force errors.
- Remaining grid-scale growth deficit: force deconvolution assumes sinc^4 while particles stay near lattice (cos^2).
- Shot noise: constant V/N subtraction; Jing (2005) correction would fix bins near Nyquist.
- Velocities currently only enter via RSD; using them as data needs a likelihood extension.
- Check AbacusSummit IC availability/normalisation for the boxes used.
- Upstream notebook bug: NUTS/HMC branch imports `flbench.flbench.samplers`.
