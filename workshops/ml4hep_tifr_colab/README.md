# ML4HEP-TIFR NSBI tutorial — Google Colab edition

These are the **Colab-ready tutorial notebooks**. Exercises 1--4 mirror the
original ML4HEP-TIFR sequence, while Exercises 5--11 extend the tutorial with
hybrid neural ratio estimation, efficient Asimov sampling, model
misspecification, semi-parametric systematic uncertainties, and dual hybrid
Bayesian posterior/likelihood estimation, followed by external `sbibm`
benchmarking and a simulator-calibrated hybrid Neyman construction. Each
notebook includes:

1. exactly one **"Open in Colab"** badge, and
2. a **setup cell** near the beginning that installs the required dependencies,
   pulls the relevant tutorial helpers, creates or reuses the legacy working
   directory, and acquires or generates the required data when they are missing.

Use these when the local `pixi` environment isn't available — no install, just a
browser.

## Open in Colab

| Notebook | |
|---|---|
| Exercise 1 — Summary statistics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_1_summary_statistics.ipynb) |
| Exercise 2.1 — Visualise the data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_1_visualize_data.ipynb) |
| Exercise 2.2a — SigvsRef training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_2a_SigvsRef_training.ipynb) |
| Exercise 2.2b — BkgvsRef training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_2b_BkgvsRef_training.ipynb) |
| Exercise 2.3 — Parameter fitting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_2_3_parameter_fitting.ipynb) |
| Exercise 3 — Parameterised CARL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_3_parameterized_carl.ipynb) |
| Exercise 4 — Normalizing flows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_4_normalizing_flows_density_estimation_direct_likelihood.ipynb) |
| Exercise 5 — Hybrid flow and density ratios | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_5_Hybrid_NormalizingFlow_DensityRatio.ipynb) |
| Exercise 6 — Neural importance-sampled Asimov data | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_6_NeuralImportanceSampling_Asimov.ipynb) |
| Exercise 7 — Asimov closure and misspecification | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iris-hep/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_7_Asimov_Misspecification_Coverage.ipynb) |
| Exercise 8 — Semi-parametric systematics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_8_SemiParametric_Systematics.ipynb) |
| Exercise 8 (FNF) — Normalization-preserving systematics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_8_SemiParametric_Systematics_FNF.ipynb) |
| Exercise 9 — Dual hNPE--hNDE | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_9_Hybrid_NPE_NDE.ipynb) |
| Exercise 10 — `sbibm` hybrid benchmark | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_10_SBIBM_Hybrid_Benchmark.ipynb) |
| Exercise 11 — Hybrid Neyman construction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/Exercise_11_Hybrid_Neyman_Construction.ipynb) |

## Notes for running on Colab

- The helper code is loaded from `workshops/ml4hep_tifr_colab`, but generated
  samples, trained models, saved densities, and plots remain in the legacy
  `workshops/ml4hep_tifr` working directory. The setup cell creates that
  directory when needed and reuses it when it already exists, including in
  Google Drive, so earlier training outputs remain available.
- **Use a GPU runtime** for the training notebooks (*Runtime → Change runtime
  type → GPU*). The setup cell installs `pytorch-lightning onnx onnxruntime
  onnxscript iminuit mplhep`; everything else (torch, jax, scikit-learn, …)
  ships with Colab. (`onnxscript` is needed by recent `torch.onnx.export`.)
- Dataset sizes are configured near the top of each setup cell. Lower them, or
  lower `number_of_epochs` / `N_TRAIN` in the training cells, for a quicker
  pass; raise them for less Monte-Carlo noise in the fit.
- Most notebooks are self-contained. `Exercise_2_3` *loads* the density-ratio
  models trained by `Exercise_2_2a` and `Exercise_2_2b`; Exercises 6--8 and 11
  reuse outputs from Exercise 5 as described below. Because each Colab notebook
  is a fresh runtime, run prerequisite notebooks first and set
  `USE_DRIVE = True` so their trained models persist to Google Drive.
- Exercises 6 and 7 load the expensive PRESEL, reference-flow, and ratio
  checkpoints produced by Exercise 5. Run Exercise 5 first and keep
  `USE_DRIVE = True` in all three notebooks so those checkpoints persist.
- Exercise 8 loads the PRESEL and nominal signal/reference and
  background/reference checkpoints produced by Exercise 5, then trains only
  the four new scale-variation ratios. It also creates the up/down parquets in
  streamed batches when the nominal generated samples already exist. Nominal
  generation is also streamed, and Exercise 8 skips the unused pseudo-data and
  generator-level plots, so the 100M/20M-event configuration has bounded peak
  memory. Keep `USE_DRIVE = True` in Exercises 5 and 8 so the nominal
  checkpoints persist. Its final Asimov diagnostic profiles both `mu` and the
  scale nuisance, reports the residual nuisance displacement from zero, and
  exports every completed figure to self-contained scripts in
  `exercise8_figures_scripts/`.
- The Exercise 8 FNF variant reuses the Exercise 5 PRESEL, reference-flow, and
  four-member signal/reference and background/reference checkpoints. It
  reconstructs the saved ONNX ratio MLPs as exact frozen PyTorch layers so the
  systematic residual can be differentiated with respect to its transformed
  coordinates; no nominal flow or density ratio is retrained. Only the
  invertible scale-deformation residuals are trained, in an
  Exercise-8-FNF-specific directory. The notebook validates them against the
  analytic selected density at trained and unseen nuisance values, checks
  normalization without an alpha-dependent rescaling, and tests
  two-dimensional Asimov closure. Its figures are exported to
  `exercise8_fnf_figures_scripts/`.
- Exercise 9 is self-contained and uses a small nonlinear Bayesian simulator
  with an explicit nuisance parameter. It trains a conditional spline NPE, a
  four-member posterior-residual ratio ensemble, and the dual spline hNDE
  likelihood with a second four-member ratio ensemble. The raw ratios are
  averaged without post-hoc calibration and use the Exercise 5 learning-rate
  sequence (`1e-3`, `1e-5`, `1e-7`, ...). Setting `FAST_MODE = True` provides
  a quicker first Colab run; the full setting increases the independent
  training and validation samples for smoother evidence and selection
  calculations. Its checkpoints are stored in the same persistent working
  directory when `USE_DRIVE = True`, and its completed figures are exported to
  self-contained scripts in `exercise9_figures_scripts/`.
- Exercise 10 is self-contained and installs the official `sbibm` task and
  metric framework. It caches an exact-budget prior-predictive simulation bank,
  trains hNPE and (for continuous tasks) the dual hNDE with optional two-fold
  cross-fitting, and compares 10,000 posterior samples to the official
  reference posteriors with the paper's five-fold C2ST. The `challenge` profile
  is intentionally a long GPU run; `quick` performs a one-observation
  diagnostic at a 10,000-simulation budget. SIR and Lotka--Volterra additionally
  require the Julia/diffeqtorch backend and are not default Colab targets.
- Exercise 11 loads the frozen Exercise 5 PRESEL, reference-flow, and
  four-member density-ratio checkpoints, together with the held-out density
  arrays saved by Exercise 5. It generates resumable, compressed
  pseudo-experiment ensembles; trains a conditional quadratic-spline model for
  the profile-likelihood-ratio statistic; adds hNDE and simulator residual
  corrections; and audits the resulting 95% critical-value curve on an
  independent simulator split following LF2I. The default full run uses
  500,000 hNDE toys, 100,000 simulator-calibration toys, and a separate
  100,000-toy audit; set `FAST_MODE = True` for a structural first pass. Run
  Exercise 5 through its hybrid-density validation first and keep
  `USE_DRIVE = True` so all required checkpoints and arrays persist.
