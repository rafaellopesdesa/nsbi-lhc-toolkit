# Unbinned calibration, step by step

This is a fresh sequence. It does not load the old Exercise 13 encoder, heads,
or training-size caches. Run the notebooks in order, using the same `RUN_NAME`
in their setup cells. A GPU is recommended for notebooks 02 and 04.

| Notebook | What it builds | Open |
|---|---|---|
| `01_Samples.ipynb` | Five-dimensional signal/background samples and detector-scale anchors; no preselection | [Colab](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/calibration/01_Samples.ipynb) |
| `02_Hybrid_Density.ipynb` | Reference flow, nominal and systematic ratios, closure plots, weighted unbinned Asimov fits | [Colab](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/calibration/02_Hybrid_Density.ipynb) |
| `03_Misspecified_Model.ipynb` | Signal contaminated by background, controlled by one mixture fraction | [Colab](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/calibration/03_Misspecified_Model.ipynb) |
| `04_Amortized_Inference.ipynb` | Population response and finite-experiment profiling learned by minimizing the likelihood; direct-fit validation | [Colab](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/calibration/04_Amortized_Inference.ipynb) |
| `05_Statistic_Flows.ipynb` | Reference distribution, reference residual, and simulator residual flows; held-out calibration checks | [Colab](https://colab.research.google.com/github/rafaellopesdesa/nsbi-lhc-toolkit/blob/ml4hep_school_tutorial/workshops/ml4hep_tifr_colab/calibration/05_Statistic_Flows.ipynb) |

The default expected yields are $S=100$ and $B=1\,000$ at $\mu=1$, so
$S/B=1/10$. The shared `TAG = "sb10_exppoly_v2_20260930"` selects a fresh
run directory in all five notebooks. Start with 01 and run through 05; the
previous `default` run is preserved. An explicit `CALIBRATION_RUN` environment
variable overrides the tagged path and is reported in the setup cell.
Simulation sample sizes are independent of these experimental yields. The
reference flow has a balanced signal/background training mixture, as in
Exercise 5: 500,000 events from each process. The nominal ratios use five
million events per class and four members; each systematic ratio uses one
million per class. These are input pool sizes before the trainer's internal
training/validation/holdout split. All six physical anchor files share row
partitions so paired systematic variations cannot cross the external holdout.

Notebook 02 currently defaults to `DIAGNOSTICS_ONLY = True`: with the existing
TAG and trained models, restart the runtime and use **Run all**. Sections 1–6
are skipped. Section 7 loads the saved networks directly and checks every
member, the arithmetic ratio ensemble, and the analytic toy ratio on fresh
independent events. Reliability residuals include approximate pointwise 95%
intervals and class counts; exact log-ratio errors, cross-entropy differences,
normalization, ESS, and tail fractions are saved under
`hybrid/ratio_diagnostics/`. This mode never trains or changes model artifacts,
normalization, or configuration, and missing checkpoints produce an error.
For a first training run or the original full closure study, set
`DIAGNOSTICS_ONLY = False`. Diagnostic samples used for development decisions
must be followed by fresh independent final validation.

The fitted nuisance uses the paper's polynomial–exponential interpolation,
with $-1\leq\alpha\leq1$. Each complete process density is divided by its
nuisance-dependent normalization; the denominator is differentiated in every
fit and neural loss. Normalization coefficients are precomputed on the fixed
reference integration bank, so expected yields stay fixed. This is a numerical
approximation to continuous normalization; notebook 02 checks another bank
throughout the nuisance interval.

The degree-six polynomial can become negative for rare extreme anchor ratios.
Only a failing event/process polynomial receives a nonnegative correction
$\lambda(x)\alpha^2(1-\alpha^2)^3$. It preserves the three anchors, the nominal
first derivative, and both derivatives at the exponential joins. Positive
polynomials retain the paper's exact interpolation. The resulting degree-eight
coefficients still permit cached normalization and sampling. Notebook 02
reports the repaired fraction and added probability mass; large corrections
would indicate a poor interpolation model. Densities are not silently clipped.

Physical toys use the continuous detector-response scale $1+0.1\alpha$.
The nominal and up/down definitions are unchanged; intermediate simulator
densities can differ from the fitted interpolation. Notebook 02 explicitly
checks that approximation at $\alpha=\pm0.5$. The nominal physical response
and calibration studies retain $\alpha_{\rm gen}=0$. Reference toys follow
the fitted normalized exp-poly law, using polynomial cumulative sums and
binary search on a finite proposal bank. Their signal contamination is applied
**after** normalizing the two process densities, keeping the mixture fraction
constant across nuisance values. All saved encoder features are good-model
anchors; `model_epsilon` belongs to the fitted likelihood configuration.

The likelihood contains the Gaussian term $-(y-\alpha)^2/2$. Toy auxiliary
observations follow the requested law $y\sim\operatorname{Uniform}[-2,2]$.
The two choices serve different purposes. Calibration and coverage in these
notebooks refer to that uniform-auxiliary ensemble, with the stated generating
nuisance, and do not establish coverage for Gaussian auxiliary observations
or uniformly over nuisance parameters.

The response network sees the generating signal strength and returns
$(g(\mu),a(\mu))$. Its objective is the physical simulator's population
negative log likelihood. The experiment networks see all unbinned ratio
features, the count, and the auxiliary observation; the conditional network
also sees the tested POI. Their objectives are negative likelihoods, not
regression to labels from numerical fits. Direct fits are reserved for
validation. Neither training objective imposes a quadratic statistic.

Evaluating the final likelihood at predicted parameters still sums over the
events. Amortization removes repeated numerical minimization; it does not make
every operation independent of event count. Approximate fits may produce a
negative raw likelihood difference. We show this diagnostic and explicitly
calibrate the nonnegative statistic $T_{\rm NN}=\max(0,T_{\rm raw})$, including
its atom at zero. Good calibration of this statistic does not demonstrate
that the neural profiling approximation is accurate; notebook 04 checks that
separately.

Notebook 04 uses 6,000 training experiments per source, 512 monitor experiments,
120 training epochs, learning-rate decay, gradient clipping, and best-checkpoint
selection. Boundary and nominal-nuisance examples supplement the continuous
training design. Training includes a conditional fit at the predicted global
POI to improve agreement between heads. Fixed training/held-out monitors use
the same query design, with separate global and conditional objective histories.
Global inference candidates are independent of the query, so negative signed
gaps remain visible rather than being removed by changing the denominator.

Diagnostics include independent normalization and positivity scans, nominal
derivative comparisons with the old linear morph, internal Asimov closure,
physical population closure at zero deliberate misspecification, response fits
on repeated independent banks, separate global/conditional NLL regrets, nuisance
residuals near zero, tail statistic errors, and raw-head/selected-minimum checks.
The independent-bank spread is reported as Monte Carlo variation, not a
confidence interval.

The three distribution models use disjoint training samples. Reference toys
are generated at $(g(\mu),a(\mu))$ and simulator toys at the physical $\mu$;
both evaluate the same frozen statistic at $\nu=g(\mu)$. The successive maps
are $Q$, $K$, and $G$, giving the calibrated CDF $G\circ K\circ Q$.

The idea of learning nuisance-aware statistics has precedents, including
[Heinrich, *Learning Optimal Test Statistics in the Presence of Nuisance
Parameters*](https://arxiv.org/abs/2203.13079). That paper's classification
construction is distinct from the direct likelihood-minimization objectives
implemented here.

For local use, install `requirements.txt`, make the repository's `src/`
available on the Python path, and launch Jupyter in this directory. Start a
fresh kernel when moving from the old exercises, which have helpers with the
same module names. Colab setup handles the source checkout, dependencies,
and optional Drive storage. Use a new `RUN_NAME` for a changed model or
training setup, and run its five notebooks in order.

Training outputs and samples live under the selected run directory and are
not committed. Full-size training and its scientific validation are intended
to be run in the notebooks; a small integration run only checks that their
stages connect and that the numerical identities hold.

For focused numerical checks, run `python -m unittest test_calibration.py` from
this directory. These checks exercise interpolation and gradients, finite-bank
closure and sampling, small inference training/checkpoint round trips, and
the three statistic-flow stages; they do not establish full-training accuracy.
The lighter `python -m unittest test_ratio_diagnostics` checks the new ratio
diagnostics, exact-density oracle, and frozen-member prediction interface
without loading trained networks or requiring a GPU.
