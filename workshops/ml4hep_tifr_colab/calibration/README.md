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

The default expected yields are $S=100$ and $B=10\,000$ at $\mu=1$.
Simulation sample sizes are independent of these experimental yields. The
reference flow has a balanced signal/background training mixture, as in
Exercise 5: 500,000 events from each process. The nominal ratios use five
million events per class and four members; each systematic ratio uses one
million per class. These are input pool sizes before the trainer's internal
training/validation/holdout split. All six physical anchor files share row
partitions so paired systematic variations cannot cross the external holdout.

The nuisance interpolates **densities** linearly between normalized nominal
and detector-scale anchors, with $-1\leq\alpha\leq1$. Expected yields stay
fixed. There is no extrapolation to negative densities. The physical toy
simulator uses the same anchor-mixture interpolation; this differs from
continuously shifting every event's detector scale between the anchors.
Normalization on a finite reference integration bank is a numerical
approximation to continuous normalization; notebook 02 checks another bank.

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
