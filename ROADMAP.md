# Roadmap

BlackPearl Morphix is developed in phases. Each phase ends in something that can be checked: tests that pass, a figure, a table in the paper, or a working demo. Every task below is sized to land as one or a few focused commits.

**Owners:** AR = Aditya Raj · SS = Senthuran Srimurugan · UR = Utkarsh Raj · SA = Sughra Azim
**Advisors:** SK = Shubham Kumar (biology, omics, bioinformatics) · PP = Prachi Prakash (research direction)

Status: ✅ done · 🔄 in progress · ⬜ planned

---

## Phase 0: Research prototype ✅

Stages 1–3 implemented as Colab notebooks, with the results written up in the paper draft.

- ✅ Permutation-equivariant flow-matching model of tissue neighbourhoods (Stage 1)
- ✅ Cell-level reconstruction of the Perturb-FISH screen; knockout signal check and conditional model (Stage 2)
- ✅ Expression-conditioned model with a leakage guard and kNN / ridge baselines (Stage 2b)
- ✅ Virtual laboratory with Bayesian hit calling and active experiment selection (Stage 3)

---

## Phase 1: Reproducible research codebase ⬜

*Goal: anyone can install the package, run one command per stage, and get the paper's numbers.*

| # | Task | Owner | Done when |
|---|---|---|---|
| 1.1 | Turn `src/` into an installable package `morphix/` with `pyproject.toml` | SS | `pip install -e .` works |
| 1.2 | Merge the duplicated `pairwise_dist`, `TimeEmbedding`, `DistanceBias`, `Block` into one `morphix/nn.py` | UR | Stage 1 and Stage 2 models import the shared layers |
| 1.3 | Config files (YAML) for every experiment instead of values hard-coded in notebooks | SS | each table row in the paper has a config |
| 1.4 | Command-line entry points: `morphix train`, `morphix sample`, `morphix evaluate` | AR | notebooks call the CLI |
| 1.5 | Fixed seeds and deterministic data splits, with split files saved to disk | SA | two runs give identical metrics |
| 1.6 | Unit tests: permutation equivariance, rotation augmentation, loss shapes, metric sanity checks | UR, SA | `pytest` passes |
| 1.7 | GitHub Actions CI running lint (`ruff`) and tests on every push | SS | green badge on the README |
| 1.8 | Data download script with checksums and a local cache (protects against data-server outages) | AR | one command fetches all datasets |

## Phase 2: Stronger evidence ⬜

*Goal: every claim in the paper is supported by multiple splits and proper statistics.*

| # | Task | Owner | Done when |
|---|---|---|---|
| 2.1 | Run Stage 2 and 2b on seeds 1–2 (three spatial splits) once Brain Image Library access resumes | SS | mean ± s.d. in Tables II–III |
| 2.2 | Bootstrap confidence intervals for all metrics | SA | intervals in every table |
| 2.3 | Ablations: distance bias on/off, model width and depth, neighbourhood size (32 / 64 / 128 cells) | UR | ablation table |
| 2.4 | Stronger leakage control: shrink cell masks or use a segmentation-aware filter; compare with the 2× rule | AR, SK | leakage analysis section |
| 2.5 | Calibration of generated distances (the known bias in Stages 2–2b) | UR | bias reduced and reported |
| 2.6 | Sensitivity of Stage 3 to effect sizes, prevalence and noise level | SA | Stage 3 robustness figure |

## Phase 3: Richer biology ⬜

*Goal: move beyond three coarse classes and one dataset, with expert input on the biology.*

| # | Task | Owner | Done when |
|---|---|---|---|
| 3.1 | Finer cell-type annotation for Perturb-FISH neighbours (T-cell subsets, stromal types) | AR, SK | annotation notebook and validation |
| 3.2 | Add a second spatial dataset (e.g. MERFISH or Xenium tumour data) | SS | Stage 1 and 2b results on both datasets |
| 3.3 | Pathway-level interpretation of Stage 2b: which gene programmes predict which neighbourhoods | AR, SK | enrichment analysis |
| 3.4 | Condition on multi-omic profiles (expression plus pathway scores), drawing on meta-omics methods | UR, SK | conditioning ablation |
| 3.5 | Review the biological claims and limitations with the advisors | all, SK, PP | review notes addressed |

## Phase 4: 3D and whole tissue ⬜

*Goal: generate tissue beyond 64-cell 2D patches.*

| # | Task | Owner | Done when |
|---|---|---|---|
| 4.1 | Extend positions to 3D, with SE(3)-equivariant attention | UR, SA | 3D model trains on a 3D dataset |
| 4.2 | Stitch patches into larger tissue with overlapping, consistent generation | SA | whole-section samples |
| 4.3 | Speed: fewer sampling steps (rectified-flow distillation) | UR | 10× faster sampling at similar quality |

## Phase 5: Experiment-design engine ⬜

*Goal: a tool a lab could use to plan a real perturbation screen.*

| # | Task | Owner | Done when |
|---|---|---|---|
| 5.1 | Package Stage 3 as a library: posterior updates, stopping rules, strategies | SS | `morphix.design` module with tests |
| 5.2 | Use the generative model's predictions as priors for active selection | AR, UR | model-informed strategy beats uninformed active selection in simulation |
| 5.3 | Batch selection (choose several knockouts per round, as real plates require) | SA | batch strategy evaluated |
| 5.4 | Cost model: cells, plates and imaging time per strategy | PP, AR | cost table for a realistic screen |
| 5.5 | Write up Phases 2–5 as a paper or workshop submission | all | submitted manuscript |

## Phase 6: Closed-loop robotic laboratory pilot ⬜

*Goal: a first real predict → select → execute → learn loop. This needs an established lab with the facilities, biosafety approval and supervision.*

| # | Task | Owner | Done when |
|---|---|---|---|
| 6.1 | Find a host lab and supervisor; agree the scope of the pilot | AR | supervisor and lab confirmed |
| 6.2 | Software interface between the selection engine and lab scheduling / robot control | SS, UR | dry run in simulation |
| 6.3 | Image-to-cell pipeline: segmentation into positions and classes, the model's input format | SA | pipeline validated on public images |
| 6.4 | Pilot with arrayed, randomised perturbations on a small gene panel | all, with host lab | first closed-loop round completed |
| 6.5 | Check that real knockout effects match the assumptions used in Stage 3 | AR, SK | comparison reported |

---

## How we work

- One focused change per commit, with a message that says what changed and why.
- New features come with tests; new results come with the config that produced them.
- Negative results are reported, as in the paper.
