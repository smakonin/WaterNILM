# Improvements, review findings, and remaining work

## Included in this update

1. Replace the two-line README with the method overview, honest scope, notebook inventory, installation/demo commands, data contract, AMPds guidance, reproducibility limits, and complete research attribution.
2. Preserve all 13 original files unchanged under `archive/`, including the original README, and explain the new layout and external notebook paths.
3. Add DOI and working author-PDF links for the water paper, SSHMM_NILM paper and AMPds2 paper; include the supplied thesis PDF and a citation with its unverified DOI explicitly noted. Add BibTeX and CITATION.cff without inventing software-author contributions or a software license.
4. Extract the core notebook workflow into `model.py`, `data.py` and a command-line entry point. Use descriptive names, docstrings, functions with explicit inputs, named units, local state and JSON model files. The core needs only Python's standard library.
5. Validate timestamps, pulse increments, states and activity intervals. Select labels by timestamp. Keep labels out of training, require held-out prediction periods and protect existing output files.
6. Provide log-domain compatibility decoding and a separately identified exact decoder. Test the extraction against notebook functions and test exact inference against exhaustive enumeration.

## Findings that affect correctness

### Label alignment and column names

`Untitled6.ipynb` cell 1 halves the electricity/water inputs and assumes DWW is already year two. The checked-in DWW spans both years and has `unix_ts,counter,avg_rate` headers, while later evaluation cells use `DWW["AVG_RATE"]`. On the checked-in files, that uppercase field access fails; merely changing capitalization would still leave the positional alignment wrong. The adapter normalizes names and selects by timestamp.

### Higher-order decoding

Cell 8 chooses one predecessor for each possible current water value. For order greater than one, the dynamic-programming state must retain the relevant history. Two paths with the same final water value can have different next-step probabilities. The exact mode fixes this; legacy mode preserves the historical pruning for comparisons. This behavior change is explicitly identified because it can change research results.

### Underflow and missing paths

Repeated probability multiplication can underflow on long sequences. The notebook then prints diagnostics and eventually takes `max()` of an empty collection. Log-space computation prevents underflow; the port reports genuine infeasibility explicitly. Log-space ties are not guaranteed to be bit-for-bit identical to product-space ties.

### Activity boundaries

The main notebook's `samples()` indexes into an empty split for all-off data. Earlier `spans()` variants omit the final span, and `switchTimes()` comments acknowledge assumptions about the first/last off periods. The new interval utility handles empty input and boundary activities; the exploratory helper variants remain in the archived notebooks.

### Hidden preprocessing and evaluation leakage

PMF extraction depends on local `Library_PMF` code and assumptions about column order. Some notebook evaluation/plotting cells use a loop index from a selected subset to remove a different activity from training (`Untitled6.ipynb` cell 11), so those plots are not a reliable leave-one-out benchmark. Recover exact data preparation and fold membership before reporting reproducibility. For new evaluations, derive PMF boundaries within each training fold.

## Integration

The modern Python workflow and documentation live at the root; `archive/` preserves the original snapshot. The corrected decoder remains distinct from compatibility decoding. Future scientific changes should remain separately documented and tested.

## Remaining work, in priority order

| Priority | Work | Completion evidence |
| --- | --- | --- |
| High | Recover and pin the original PMF/state extraction dependency and data release. | Known source revision and saved boundaries; verify state IDs against notebook outputs. |
| High | Reproduce the 185-activity, ten-fold dishwasher experiment. | Stored activity timestamps/folds; verified DWW alignment; original metrics and separate legacy/exact results. |
| High | Confirm repository software and annotation licensing with the rights holders. | Explicit license files and attribution terms; do not reuse a paper/dataset license for code by inference. |
| Medium | Turn figure notebooks into short, named examples importing package functions. | Restart-and-run-all success in a pinned environment, figures checked against documented inputs. |
| Medium | Add descriptive, maintained example notebooks; retain the original `archive/` snapshot. | New examples run from a clean kernel and refer to the preserved originals. |
| Medium | Add CI on the supported Python versions. | Unit tests and a packaged CLI smoke test run automatically on proposed changes. |
| Medium | Benchmark memory and runtime on full AMPds inputs; stream CSV preparation if needed. | Measured resource requirements and unchanged timestamp/label alignment. |
| Later | Add a validated SSHMM_NILM integration and jointly constrained multi-appliance estimation. | Explicit input/output contract, propagated NILM error analysis and conservation checks. |

The archive move preserves original blobs rather than rewriting notebooks. Historical notebook names in these findings now refer to `archive/`.
