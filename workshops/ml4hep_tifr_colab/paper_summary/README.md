# SLCP paper comparison: JANA and hybrid residual correction

This directory turns the exploratory Exercise 9 notebooks into a paired,
paper-oriented SLCP study.  It is intentionally self-contained: notebooks
import only modules stored here.  The exact-JANA compatibility layer in
`utils_jana.py` is launched by `utils.py` in an
isolated legacy environment because the paper's TensorFlow/BayesFlow pins are
not compatible with the modern PyTorch environment used by the matched flows
and residual classifiers.

## Scientific question

At each simulator budget, we ask whether a residual density-ratio correction
improves two conditional density estimators when the simulator calls, trained
flows, observations, evaluation samples, and ML seeds are paired.  The main
comparison contains six rows grouped into two controlled comparisons:

1. **JANA-paper**: the algorithm and hyperparameters from the pinned JANA
   source, adapted only to consume the pre-generated SLCP bank;
2. **JANA + multiclass correction**: the same exact JANA checkpoints with one
   equal-prior three-class CE correction;
3. **JANA + binary corrections**: those checkpoints with two independent
   equal-prior binary CE corrections;
4. **separate flows**: independently trained posterior and likelihood
   ensembles selected by validation NLL;
5. **separate flows + multiclass correction**: those exact flow checkpoints,
   the broadened/defensive proposal, and one three-class correction; and
6. **separate flows + binary corrections**: the same checkpoints and proposal
   with two binary corrections.

The two causal contrasts are within-base: uncorrected JANA versus corrected
JANA, and uncorrected separate flows versus corrected separate flows.  No
claim about correction quality is inferred from a cross-base comparison.
Multiclass and binary corrections are also compared directly with paired
seed-and-observation contrasts within each base.

No SLCP sign symmetry is supplied to any primary method.  A symmetry-aware
study may be added later as a separately labelled appendix test, but it is not
part of this campaign.

## Simulation accounting

`00_prepare_SLCP_samples.ipynb` generates one deterministic master bank of one
million prior-predictive pairs.  The budgets are nested prefixes,

```text
D_10k  subset of  D_100k  subset of  D_1M,
```

and one fixed master-row assignment determines training and validation
membership.  Therefore both parts of the split are themselves nested across
prefixes.  At a fixed budget, the separate posterior flow, likelihood flow,
and every ratio estimator use the same genuine training pairs.  Reusing an
event in several estimators does not incur another simulator call; generated
proposal examples also do not count toward the simulation budget.

There is one explicitly reported exception for the external reproduction row.
The exact upstream JANA benchmark consumes two rows for framework shape
inference and its `Trainer` consumes a second two-row consistency/ActNorm
pilot; training then uses **N simulations plus a separate fixed 300-simulation
validation bank**.  We retain all of those calls, so tables report its
simulator cost as `N + 2 + 2 + 300 = N + 304`, not `N`.  The same shape, pilot,
and validation rows are reused for all exact-JANA budgets and seeds.
They do not enter the separate-flow fits, proposal selection, or final audit.
This fixed overhead is one reason JANA-paper is treated as an external
reproduction/scaling baseline rather than the causal control for the ratio
correction.

A fifth, separately seeded audit bank is never used for fitting, architecture
or proposal selection, early stopping, or checkpoint selection.  Official
SBIBM reference posterior samples and observations are evaluation resources,
not training simulations.  Manifests record all five bank roles and seeds,
row indices, array hashes, configuration signature, and consumer roles so this
accounting can be audited.

## Flow and proposal protocol

Capacity is screened with one conditional RQS flow per architecture and ML
seed, then validation NLL is aggregated across the preregistered seeds.  For
each budget and route separately, the selected model is the smallest candidate
within one standard error of the best mean validation NLL.  The selected
architecture is then trained as the deployed four-member ensemble; all four
members see the complete training partition and differ only through
initialization and minibatch order.  Every fixed learning-rate schedule runs
to completion and deploys its final epoch; validation does not select an
earlier epoch.  This screening/deployment distinction is recorded in the
manifests.  The preregistered grid is

```text
coupling blocks: 4, 6, 8
hidden width:    32, 64, 128
```

with the remaining transform and training choices fixed in `config.py`.
Posterior and likelihood routes may select different sizes from the same grid.
Both matched flows and residual classifiers use six equal-duration learning-
rate plateaus at `1e-4`, `1e-5`, `1e-6`, `1e-7`, `1e-8`, and `1e-9`; every
declared decade therefore receives optimization steps.

The hybrid does not train another flow.  If `T` is a selected learned
transport, its proposal samples the latent variable from
`Normal(0, tau^2 I)` rather than mixing nominal and broad latent bases.  In
parameter space the deployed proposal is

```text
g_phi(theta | x) = (1 - epsilon) q_phi,tau(theta | x)
                 + epsilon p(theta).
```

For `q_eta(x | theta)`, latent broadening is used without a prior component,
because the parameter prior is not a distribution over observations.  The
`tau` (1.25 or 1.5) and `epsilon` (0.05 or 0.10) choice is selected once at
100k using proposal closure and weight-efficiency criteria that do not inspect
the official reference posterior, then frozen across the three headline
budgets.  The flow-validation partition is split once into disjoint pilot
training, checkpoint-validation, and untouched closure thirds.  Selection
uses the worst route and worst correction factorization, so neither multiclass
nor binary correction receives a tuning advantage.  A 100k ablation compares
nominal, broad-only, and broad-plus-prior proposals.

All residual models are plain ReLU MLP ensembles optimized with equal-prior
cross entropy and Adam.  There is no dropout, weight decay, layer
normalization, calibration loss, bridge loss, or normalization penalty.
Ratios remain in log space through evaluation and ensemble combination; CE
training contains no exponentiation.  A deterministic shuffled-cycle sampler
guarantees that every genuine training row is consumed before any row is
reused.  Normalized weights exponentiate only after a maximum subtraction.

## Notebook order

Run the notebooks in this order:

1. `00_prepare_SLCP_samples.ipynb` — generate and fingerprint the nested master
   bank, exact-JANA two-row shape bank, two-row Trainer pilot, 300-row
   validation bank, and independent audit bank;
2. `01_SLCP_flow_capacity.ipynb` — train the capacity screens, apply the
   one-standard-error rule, and save the selected matched checkpoints;
3. `02_SLCP_JANA.ipynb` — run the pinned JANA-paper baseline and evaluate the
   nominal selected ensembles on posterior and likelihood routes;
4. `03_SLCP_hybrid.ipynb` — load those exact selected checkpoints, select the
   proposal on validation data, and train multiclass and binary corrections;
5. `04_SLCP_comparison.ipynb` — read completed artifacts only and produce
   paired tables and paper figures.

Each notebook exposes `PROFILE`, `BUDGETS_TO_RUN`, `ML_SEEDS_TO_RUN`, and
`LOAD_IF_AVAILABLE` in its configuration cell.  `PAPER` is the scientific
campaign.  `SMOKE` uses tiny synthetic budgets and one observation solely to
exercise the full code path; its outputs are never paper results.

The full PAPER campaign is intentionally larger than one ordinary Colab
session, especially the literal JANA 1M run (100 epochs at batch size 32).
Shard execution without changing the campaign signature by setting, for
example, `PAPER_SUMMARY_RUN_BUDGETS=1000000` and
`PAPER_SUMMARY_RUN_SEEDS=31082026`.  Capacity screens and all method results
write per-budget/per-seed atomic shards and aggregate when later notebooks are
rerun.  Notebook 04 refuses to label a PAPER campaign complete until every
registered budget, seed, observation, method, and diagnostic is present.

## Diagnostics and outputs

All six methods are evaluated per observation, budget, and ML seed.  The
standard result schema includes:

- C2ST and MMD against the official posterior for both inference routes;
- posterior-route versus likelihood-route C2ST and MMD;
- posterior-predictive `x` and joint `(theta, x)` C2ST;
- importance ESS, maximum normalized weight, and weight-tail summaries;
- conditional likelihood normalization and Bayes-cycle diagnostics;
- flow validation NLL, classifier train/validation CE, ensemble spread, and
  fresh-bank closure; and
- an SLCP exact-likelihood and Bayes-cycle audit on one fixed independent
  `(theta, x)` table shared by all six methods; method-native proposal
  diagnostics are retained separately.

Means and uncertainties across ML seeds are reported alongside every
observation-level result.  Observation 7 is never hidden by an across-
observation average.  Paired tables retain raw treatment-minus-control
differences and add an oriented improvement whose positive sign always means
closer to the preregistered metric goal (for example, `|C2ST - 0.5|` decreases).

By default Colab writes large products to
`MyDrive/hybrid_nsbi_ml/paper_summary_SLCP`.  Locally, set
`PAPER_SUMMARY_ARTIFACT_ROOT` to an explicit persistent directory.  Runtime
products are ignored by Git.  Complete stages write manifests, long-form CSV
tables, sample archives, and PNG/PDF figures beneath the artifact root.

The exact baseline runs in an isolated Python 3.11 environment.  Notebooks 02
and 03 can create it from `requirements_jana.txt`.  In Colab the environment
is runtime-local at `/content/paper_summary_jana_env` while results remain on
Drive.  If this managed environment is left incomplete by an interrupted
installation, rerunning the environment cell rebuilds it from scratch.  An
existing environment may instead be selected with
`PAPER_SUMMARY_JANA_ENV` or `PAPER_SUMMARY_JANA_PYTHON`.

## Maintaining the notebooks

The five notebooks are generated files.  Edit `generate_notebooks.py`, then
run:

```bash
python workshops/ml4hep_tifr_colab/paper_summary/generate_notebooks.py
```

Pass one or more notebook filenames to regenerate only those files, for
example `python generate_notebooks.py 02_SLCP_JANA.ipynb 03_SLCP_hybrid.ipynb`.
This is useful when executed results in another stage must remain untouched.

The generator emits stable cell IDs and clean notebooks with no execution
counts or outputs.  Running it twice must leave the files byte-identical.
