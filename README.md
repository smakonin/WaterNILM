# WaterNILM

**Appliance water disaggregation using electrical activity and aggregate household water measurements.**

WaterNILM contains the research notebooks associated with Bradley Ellert's 2015 MSc thesis and the paper by **Bradley Ellert, Stephen Makonin, and Fred Popowich**, *Appliance Water Disaggregation via Non-intrusive Load Monitoring (NILM)*. The method learns relationships between an appliance's electrical states and whole-house water readings, then estimates the appliance's water use with a higher-order hidden Markov model (HMM) and **Capped Viterbi** decoding.

This repository is related to [SSHMM_NILM](https://github.com/smakonin/SSHMM_NILM), which provides the electricity-disaggregation foundation. **Capped Viterbi for water disaggregation and Sparse Viterbi for electricity disaggregation are distinct methods and should be cited separately.** See [Attribution and references](#attribution-and-references).

## Status and scope

The original notebooks are a research archive, with exploratory code, saved figures, local file paths, and external dependencies. The `waternilm/` package is a Python 3 extraction of the core workflow from `archive/Untitled6.ipynb`. It adds explicit inputs, validation, a command-line interface, and tests while retaining the original notebooks as references.

- The package trains from aggregate water and appliance electrical states, and predicts water use for **one appliance at a time**.
- It supports model orders 1–4, matching the range studied in the paper.
- It includes `exact` and `legacy` decoder modes; their differences are documented below.
- It runs without Jupyter or third-party Python dependencies. The historical notebooks have additional dependencies.
- It has been tested against synthetic cases and extracted notebook functions. **The published AMPds experiments have not been reproduced with this package.**

This is not a general fixture classifier: it does not identify showers, taps, toilets, or leaks. Estimates from separate appliance models are not jointly constrained; summing them can exceed aggregate consumption. The paper's electricity input was derived from submetered ground-truth current for evaluation, so its reported water results should not be described as an end-to-end benchmark using imperfect NILM predictions.

## How the method works

1. Identify appliance activity intervals from electrical measurements. The main notebook uses nonzero apparent power (`S`) for this step.
2. Quantize appliance current (`I`) into electrical states using probability-mass-function (PMF) boundaries. Electrical state `0` denotes off.
3. Express whole-house average water flow as discrete water states, normally in **0.5 L/min** increments for the second-year experiment.
4. Learn transitions between water histories and emissions linking water histories to electrical-state histories. Training uses whole-house water readings, not hand-labelled appliance water.
5. Decode each appliance activity. At every minute, the predicted water state must be no greater than the observed whole-house water state: this is the “cap.”
6. Use the hand-labelled dishwasher series only to evaluate predictions.

Each activity begins with dummy zero states. For order `n`, the model uses `n` preceding water states for transitions, and length-`n` water and electrical histories for emissions. It is not interchangeable with an off-the-shelf first-order Gaussian HMM.

The notebook's emission smoothing replaces counts of zero and one with their mean, using one if all those counts are zero, and then normalizes. The port retains this rule and adds zero-state transitions as the notebook does. It does not silently replace it with Laplace smoothing.

## Repository guide

| File or directory | Purpose |
| --- | --- |
| `archive/Untitled0.ipynb` | Initial current/water exploration, histograms, activity detection, and external NILM trials. |
| `archive/Untitled1.ipynb` | AMPds electricity export and calls to external load-disaggregation code. |
| `archive/Untitled2.ipynb` | Correlations, PMF discretization, and state-duration exploration. |
| `archive/Untitled3.ipynb` | Further electrical-state, event, and water-pattern exploration. |
| `archive/Untitled4.ipynb` | Activity clustering, dynamic time warping, and model experiments. |
| `archive/Untitled5.ipynb` | Early HMM and Viterbi experiments. |
| `archive/Untitled6.ipynb` | Main training, Capped Viterbi, evaluation, and figure experiments; source of the Python core. |
| `archive/Untitled7.ipynb`, `archive/Untitled8.ipynb`, `archive/Untitled9.ipynb` | Dishwasher annotation work for different portions of the dataset. |
| `archive/DWW_label.ipynb` | Consolidated hand-annotation workflow and export of `archive/DWW.csv`. |
| `archive/DWW.csv` | Timestamped dishwasher water annotations; not a physical dishwasher water submeter. |
| `archive/README.md` | Original two-line README, preserved unchanged. |
| `waternilm/model.py` | Activity intervals, model training, JSON model representation, and decoding. |
| `waternilm/data.py` | Named CSV fields, timestamp alignment, and AMPds preparation. |
| `waternilm/__main__.py` | `prepare`, `train`, and `predict` commands. |
| `examples/` | Small synthetic training/test files for a runnable demonstration. |
| `tests/` | Notebook comparisons, exhaustive decoding checks, and data/CLI tests. |
| `docs/PROVENANCE.md` | Source commit, code mapping, reference provenance, and validation limits. |
| `docs/PROPOSED_CHANGES.md` | Review findings and staged recommendations. |

## Original research archive

All **13 original repository files** are preserved unchanged in [`archive/`](archive/): the eleven notebooks, `DWW.csv`, and the original two-line `README.md`. The folder name identifies a historical research snapshot; the maintained Python package and documentation live at the repository root.

The move preserves file contents, saved notebook outputs, annotations, and Git history. `archive/README.md` is the original README, not a replacement archive guide. The archived notebooks are historical references and are not guaranteed to run unchanged. Their relative paths still assume an external `../AMPds/` and `../Disagg/` layout; when running from `archive/`, those resolve inside the repository root. Configure the external data/code paths in a working copy as needed. `./DWW.csv` remains beside the notebooks.

For the Python workflow, supply the labels explicitly as `--labels archive/DWW.csv` (or set `DWW_CSV=archive/DWW.csv` in the AMPds example). No dataset or notebook is silently relocated at runtime. The original snapshot is also available at [commit a76be1e](https://github.com/smakonin/WaterNILM/tree/a76be1e82920732e0cd259741b11a375cccfce10).

## Quick start: standalone Python

Use **Python 3.10 or newer**. From the repository root:

```bash
python3 -m waternilm --help
mkdir -p runs

python3 -m waternilm train examples/train.csv \
  --num-observed 3 --order 2 --model runs/demo-model.json

python3 -m waternilm predict examples/test.csv \
  --model runs/demo-model.json --decoder exact \
  --output runs/demo-predictions.csv

python3 -m unittest discover -s tests -v
```

The demonstration uses deliberately simple **synthetic data**, not AMPds observations. Its zero error is a functional check, not a research result. Commands refuse to overwrite existing files; use a new output filename when repeating an experiment.

Optional installation adds the `waternilm` command:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
waternilm --help
```

Model files are JSON. Prediction produces a CSV with timestamps, predicted water flow, aggregate flow, and optional evaluation labels. If labels are supplied, stdout reports MAE, MSE, and RMSE for all input minutes and separately for activity minutes. MSE has units `(L/min)^2`; MAE and RMSE have units `L/min`. No ground-truth labels are used during fitting or decoding.

## Input contract

The core workflow accepts a headered CSV:

```csv
timestamp,electrical_state,water_lpm,active,truth_lpm
1700000000,0,0,0,0
1700000060,1,0.5,1,0.5
1700000120,2,1,1,1
```

| Field | Meaning |
| --- | --- |
| `timestamp` (or `unix_ts`) | Unix seconds, unique and increasing at exactly 60-second intervals. |
| `electrical_state` | Nonnegative integer appliance electrical-state ID. Use the same state mapping for training and prediction. |
| `water_lpm` | Nonnegative finite whole-house average water flow in litres/minute. |
| `active` | `0` or `1`; activity indicator derived from appliance electricity. Inactive samples require electrical state zero. |
| `truth_lpm` | Optional hand-labelled appliance flow, used only for evaluation. |

`--num-observed` is the complete electrical-state vocabulary size, including off state zero. Missing minutes, duplicate timestamps, invalid states, and non-finite or negative readings are rejected rather than silently aligned or filled.

`--water-step` defaults to `0.5`. Values must lie on that grid, within a small numerical tolerance. The original notebook truncated `flow / step`; the package rejects off-grid readings to expose incompatible units or pulse regimes. Do not mix gallon-pulse and half-litre-pulse periods in one model without a documented conversion. `--max-gap` defaults to zero; increasing it joins short inactive gaps and changes activity segmentation.

Use separate training and prediction periods, split at complete activity boundaries. The CLI rejects prediction timestamps overlapping the model's training range. It cannot detect leakage from electrical-state boundaries learned on test data: fit and freeze those boundaries using training data only.

## Using AMPds and the supplied annotations

The repository does **not** bundle the electricity and whole-house water files required by the notebooks. Obtain the appropriate AMPds release from its [dataset record](https://doi.org/10.7910/DVN/FIE0S4), consult the [AMPds2 paper](https://doi.org/10.1038/sdata.2016.37), and record the exact release and preprocessing used. Historical `../AMPds/` paths and modern dataset layouts may differ.

The adapter reads these fields by name, case-insensitively:

- Appliance electricity: `TIMESTAMP` or `unix_ts`, `I` (current in A), and `S` (apparent power in VA). Use dishwasher `DWE.csv` or clothes-washer `CWE.csv` as appropriate.
- Whole-house water: `TIMESTAMP` or `unix_ts`, and `AVG_RATE` (L/min), normally `WHW.csv`.
- Optional dishwasher labels: the repository's `archive/DWW.csv`, with `unix_ts,counter,avg_rate`. `counter` is cumulative annotated volume; evaluation uses `avg_rate`.

**Alignment matters:** the checked-in `archive/DWW.csv` contains **1,051,200 one-minute rows spanning two years**, starting at Unix timestamp `1333263600` and ending at `1396335540`. `archive/Untitled6.ipynb` slices the electricity and water series to their second halves but comments that DWW is “already only second year.” That assumption does not match this checked-in file. The package selects labels by timestamp, never by matching row offsets.

For the second-year period, use inclusive start `1364799600` and exclusive end `1396335600`. To prepare it after obtaining and documenting PMF thresholds from training data:

```bash
# Set DWE_CSV and WHW_CSV to the external meter files.
DWW_CSV=archive/DWW.csv
# CURRENT_BOUNDARIES must contain your training-derived thresholds in A,
# separated by spaces. It must not be left unset.
python3 -m waternilm prepare \
  --electricity "$DWE_CSV" --water "$WHW_CSV" --labels "$DWW_CSV" \
  --boundaries ${CURRENT_BOUNDARIES:?Set training-derived current thresholds} \
  --start 1364799600 --end 1396335600 \
  --output runs/dishwasher-year2.csv
```

This prepares data; it does not create train/test folds. For a held-out experiment, prepare disjoint date ranges or split the prepared file at activity boundaries. Apply the same thresholds to both sets, then use `train` and `predict` as above. Use DWW labels only for the dishwasher; the repository does not provide equivalent clothes-washer water ground truth.

The Python adapter takes explicit thresholds and implements the corresponding bin assignment. **Automatic PMF boundary discovery from `Library_PMF.create_pmfs` has not been ported.** Alternatively, feed electrical-state IDs produced by a separately validated NILM pipeline directly into the canonical CSV. Document which route was used.

## Exact and compatibility decoding

| Mode | Behavior | Intended use |
| --- | --- | --- |
| `exact` (default) | Retains the best path for every reachable full hidden-state history. | Correct higher-order dynamic programming for the learned model. |
| `legacy` | Retains one winning history for each current water state, following the notebook's pruning. | Comparing the port with historical notebook behavior. |

For orders above one, different histories ending in the same current water state can have different future probabilities. Discarding all but one can lose the optimal path; a regression test demonstrates this. Changing this behavior is an **algorithmic correction**, not merely a readability refactor, and can change reported results.

Both modes use log probabilities to avoid underflow and backpointers to reconstruct paths. Compatibility comparisons pass on short test cases, but floating-point ties or sequences that underflowed in the original may differ. Neither mode guarantees a path for arbitrary unseen data: an impossible path raises an explanatory error. Exact decoding can retain many more histories and become expensive at higher orders. Training also has a guard against excessively large emission smoothing tables.

## Historical notebooks and reproducibility

The notebooks use NumPy, Matplotlib, scikit-learn, and external modules from a local `../Disagg/` checkout, including `Library_PMF`. Exploratory notebooks also reference SciPy, DTW packages, `hmmlearn`, `seqlearn`, and other experiments. There is no verified historical environment lockfile, and installing all current versions does not establish compatibility.

Direct execution needs additional work: notebook magics and directory changes, Python-version-sensitive division, removed plotting APIs, hard-coded absolute paths, missing intermediate files, and data-version assumptions remain in the archive. The Python package extracts the core computation; it does not run every exploratory cell or regenerate all paper figures.

The paper reports 185 second-year dishwasher activities evaluated with ten folds. A faithful reproduction should recover the same inputs, activity extraction, PMF boundaries, fold membership, label alignment, smoothing, decoder, and metric aggregation. Report compatibility and corrected results separately. Published accuracy numbers are not acceptance tests for a refactor until these choices have been recovered.

## Attribution and references

Please cite the water-disaggregation paper and thesis when using this method; additionally cite the electricity method when using SSHMM_NILM, and the dataset publication/release when using AMPds. Research authorship, software contributions, and dataset ownership are separate forms of attribution.

### Water-disaggregation method

**Bradley Ellert, Stephen Makonin, and Fred Popowich.** “Appliance Water Disaggregation via Non-intrusive Load Monitoring (NILM).” In *Smart City 360°*, LNICST **166**, pp. **455–467**, Springer, **2016**. Presented at BigDASC **2015**; the supplied accepted manuscript carries the 2015 conference information, while the publisher's chapter is dated 2016.

- DOI: [10.1007/978-3-319-33681-7_38](https://doi.org/10.1007/978-3-319-33681-7_38)
- [Download author manuscript (PDF)](https://makonin.com/Publications/papers/ellert2015appliance.pdf)
- [Publisher record](https://link.springer.com/chapter/10.1007/978-3-319-33681-7_38)

### Thesis

**Bradley Ellert.** *Leveraging Submetered Electricity Loads to Disaggregate Household Water-use*. MSc thesis, School of Computing Science, Simon Fraser University, **2015**. Defended **17 August 2015**. Senior supervisor: **Fred Popowich**; supervisor: **Wolfgang Stuerzlinger**. The thesis acknowledges Stephen Makonin's foundational research and guidance on AMPds.

- [Download thesis (PDF, supplied copy)](docs/papers/Thesis_BEllert_MSc.pdf)
- **DOI: not verified.** No DOI is given in the supplied thesis, and none was established in this review. Do not substitute the conference paper's DOI for the thesis.

### Electricity-disaggregation foundation: SSHMM_NILM

**Stephen Makonin, Fred Popowich, Ivan V. Bajić, Bob Gill, and Lyn Bartram.** “Exploiting HMM Sparsity to Perform Online Real-Time Nonintrusive Load Monitoring.” *IEEE Transactions on Smart Grid* **7**(6), pp. **2575–2585**, **2016**.

- DOI: [10.1109/TSG.2015.2494592](https://doi.org/10.1109/TSG.2015.2494592)
- [Download author manuscript (PDF)](https://makonin.com/Publications/papers/makonin2015exploiting.pdf)
- [SSHMM_NILM source repository](https://github.com/smakonin/SSHMM_NILM)

The DOI contains 2015; the final volume/issue citation is 2016. The earlier water paper cites the 2014 sparse-matrix workshop work; the journal article above is the supplied reference for SSHMM_NILM, not a claim that it originally appeared in that paper's bibliography.

### Dataset and water annotations

**Stephen Makonin, Bradley Ellert, Ivan V. Bajić, and Fred Popowich.** “Electricity, water, and natural gas consumption of a residential house in Canada from 2012 to 2014.” *Scientific Data* **3**, article **160037**, **2016**.

- DOI: [10.1038/sdata.2016.37](https://doi.org/10.1038/sdata.2016.37)
- [Download paper (PDF)](https://makonin.com/Publications/papers/makonin2016electricity.pdf)
- [Dataset record and downloads](https://doi.org/10.7910/DVN/FIE0S4)

The dataset publication credits Stephen Makonin with leading dataset development and Bradley Ellert with water annotations. DWW is derived, manually annotated reference data, with some inherently ambiguous cases; it must not be described as measured dishwasher submeter ground truth.

Machine-readable references are in [references.bib](references.bib) and [CITATION.cff](CITATION.cff). The original research acknowledges NSERC and SFU graduate awards; those acknowledgments refer to the original research, not funding for this code update.

## License and contributions

The WaterNILM project software is licensed under the **BSD 3-Clause License** ([LICENSE](LICENSE); SPDX identifier: `BSD-3-Clause`). This permits academic and commercial use, modification, and redistribution subject to retaining the required notices and disclaimer. The authors' and contributors' names may not be used to imply endorsement without prior written permission.

The software license applies to the project's Python code and original notebook code, except where separate terms or third-party notices apply. It does **not** relicense:

- **Research data and annotations**, including `archive/DWW.csv` and externally obtained AMPds data. Their applicable dataset and annotation terms remain in effect.
- **Papers and theses**, including `docs/papers/Thesis_BEllert_MSc.pdf`. These retain their existing copyright and publication terms.
- **Third-party code or dependencies**. Their respective licenses and notices continue to apply.

Please cite the relevant research using the references above and `CITATION.cff`. Scholarly citation is requested separately from the BSD license conditions.

When contributing, keep scientific changes separate from structural cleanup, retain attribution, document preprocessing and units, and add tests for behavior changes. See [the review findings and remaining work](docs/PROPOSED_CHANGES.md) for the remaining reproducibility and maintenance work.
