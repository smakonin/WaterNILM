# Source provenance and validation

## Upstream snapshot

Reviewed repository: [smakonin/WaterNILM](https://github.com/smakonin/WaterNILM).

Reviewed `master` commit: [`a76be1e82920732e0cd259741b11a375cccfce10`](https://github.com/smakonin/WaterNILM/commit/a76be1e82920732e0cd259741b11a375cccfce10), dated 2020-12-13, with commit author Stephen Makonin and message “initial version.” Commit authorship is not treated as proof of sole authorship of the historical notebooks.

The reviewed snapshot root contains DWW.csv, eleven notebooks, and the original two-line README. No AGENTS.md, package metadata, test suite, or repository-level license was present in that snapshot. All 13 original files are preserved byte-for-byte under `archive/`, including the original README. The archived blob IDs are checked against the upstream snapshot before merging.

Repository content was read through the connected GitHub account. Large notebooks were parsed as JSON, extracting source cells without executing their saved exploratory code or notebook commands. Cells below are **zero-indexed** in the first worksheet of the version-3 notebook.

| Proposed code | Source | Treatment |
| --- | --- | --- |
| `model.activity_ranges` | `Untitled6.ipynb`, cell 5, `samples` | Clearer name; empty/all-off handling and explicit gap validation. |
| `model.train` | `Untitled6.ipynb`, cell 7, `train` | Preserve counts, dummy zeros, zero-transition support and singleton smoothing; remove unused/overwritten `add` argument and commented experiments; validate inputs. |
| `model.capped_viterbi(mode="legacy")` | `Untitled6.ipynb`, cell 8, `viterbi` | Preserve per-current-state pruning; use log scores, sparse lookups and backpointers. |
| `model.capped_viterbi(mode="exact")` | Same model and Capped Viterbi formulation | Correct higher-order state retention; separately exposed behavior change. |
| `data.prepare_ampds` | Notebook cells 1 and 4; `DWW_label.ipynb` export | New timestamp-based adapter. Explicit external boundaries replace the hidden dependency on local PMF code; automatic PMF discovery is not implemented. |
| CLI and model JSON | New supporting code | Explicit train/predict separation, output paths, model metadata and error reporting. |
| `tests/notebook_reference.py` | `Untitled6.ipynb`, cells 7–8 | Extracted function bodies, comments removed by `ast.unparse`, used as comparison oracles. No new license asserted for upstream source. |

`Untitled6.ipynb` blob: `602b385dac18c0cf3a01d7da4da88f62296da717`.
`DWW_label.ipynb` blob: `bea2bb4fd9e6036d822b98e570af26a8497e7dea`.
`DWW.csv` blob: `8fa9da33b2326597a3d8b1b277f1e57dd2f65140`.

## Archive layout

`archive/` mirrors the original repository root. New code and documentation are at the current root. Original notebook filenames in the source mapping above refer to `archive/` in the current tree. Moving notebooks preserves their contents but changes where external relative paths resolve; see the root README before executing a working copy.

## Bibliographic provenance

- The supplied thesis PDF establishes its title, author, degree, institution, 2015 date, committee and acknowledgments. It contains no stated DOI. No DOI or accessible institutional download was established during this review; the unmodified supplied copy is included at `docs/papers/Thesis_BEllert_MSc.pdf` for the requested download link.
- The supplied `BigDASC_2015.pdf` is an accepted manuscript. The [Springer record](https://link.springer.com/chapter/10.1007/978-3-319-33681-7_38) establishes the final 2016 publication year, authors, volume, pages and DOI. The author's bibliography still uses 2015 in its key/year; this update uses the publisher's year and explains the conference/manuscript distinction.
- The SSHMM_NILM DOI was supplied by the requester. Author metadata and the publication bibliography corroborate the five authors and final 2016 volume/issue citation.
- Current author-manuscript links were recovered from [the author's bibliography](https://makonin.com/Publications/publications.bib). The three `Publications/papers/` PDF links in the README returned HTTP 200, PDF content types and PDF headers on 2026-09-27. The former `makonin.com/doc/` links returned 404 and were not used.
- The [AMPds2 article](https://www.nature.com/articles/sdata201637) distinguishes manually annotated DWW data from meter observations and credits Bradley Ellert's annotation contribution.

No authorship, ownership or license grant is inferred from a PDF being available to download. The supplied thesis is unchanged and retains its existing copyright notice.

## What was verified

The unit tests exercise:

1. Exact equality of training transition/emission probabilities with the notebook for orders 1–4.
2. Compatibility-decoder equality with the notebook on 40 short deterministic test sequences across those four orders.
3. Exact decoding versus exhaustive enumeration on 40 small capped problems across orders 1–4.
4. A counterexample in which historical higher-order pruning loses the optimal path.
5. A 2,000-sample sequence that would underflow with repeated small probability products.
6. Empty activity inputs, sequence edges, invalid states, infeasible paths, model serialization and prediction without model mutation.
7. Timestamp-based label selection when the label file spans a longer period than selected meter rows; gap/duplicate/missing-timestamp checks and pulse-grid validation.
8. CLI training, prediction under both decoders, synthetic metric output, overwrite protection and train/test timestamp overlap rejection.

The actual upstream DWW.csv was inspected: 1,051,200 rows; timestamps 1333263600 through 1396335540; the second half starts at 1364799600. Its second-year annotation values lie on the half-litre grid. This supports the alignment finding but does not verify all manual labels.

## Limits

Validation used Python 3.12 from the bundled runtime. Other supported Python versions have not been executed here. Full AMPds electricity/whole-house water files, original PMF environment, paper fold membership, and original figure-generation environment were not available locally, so no end-to-end scientific reproduction or full-scale runtime/memory benchmark is claimed. The adapter currently reads CSVs into memory; annual meter files can require substantial memory.

The command-line evaluation is a held-out-period workflow, not the paper's ten-fold experiment. It reports MAE/MSE/RMSE, not every historical metric. The synthetic demo is intentionally simple. The exact decoder is an explicitly changed algorithm and must be reported separately from compatibility results. A regression suite cannot establish the paper's reported accuracy.
