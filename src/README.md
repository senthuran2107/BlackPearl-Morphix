# Source modules

These are the same code blocks that the notebooks in `../notebooks/` contain, kept as plain Python files so they are easy to read and review. The notebooks are self-contained; you do not need to import these files to run anything.

- `stage1/`: `cells_data.py` (dataset loading, spatial block split, 64-cell patches), `cells_model.py` (permutation-equivariant flow-matching network, loss, Euler sampler), `cells_metrics.py` (Wasserstein / TVD metrics and baselines).
- `stage2/`: `s2_*` (focal-cell neighbourhoods, knockout-conditioned model, raw and location-matched evaluation) and `s2b_*` (expression loading and leakage filter, expression-conditioned model, per-cell evaluation). `s2b_model.py` builds on the blocks defined in `s2_model.py`.
- `stage3/`: `s3_sim.py` (virtual laboratory on real tissue, Bayesian hit calling, round-robin / random / active strategies).
