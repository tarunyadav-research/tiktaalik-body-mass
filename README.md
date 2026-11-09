# Body mass of *Tiktaalik roseae*

Code and data for:

> Yadav, T. (2026). How heavy was *Tiktaalik roseae*? A body-mass estimate tested on living fishes, with explicit uncertainty, and why specimen length is the open question.*Preprints* 2026, 202611.0854. https://doi.org/10.20944/preprints202611.0854.v1 (not peer reviewed)

Author: Tarun Yadav, Independent Researcher, Delhi, India · ORCID [0009-0000-2291-5146](https://orcid.org/0009-0000-2291-5146)

[![DOI](https://zenodo.org/badge/1442952727.svg)](https://doi.org/10.5281/zenodo.23885425)

## What it does

Estimates the body mass of *Tiktaalik roseae* specimens by Monte Carlo simulation, with explicit uncertainty.

- **Method B (primary, volumetric):** M = C · L · w · d = C · (w/L)² · (d/w) · L³.
  The width and depth come from the published reconstruction of the holotype NUFV 108 (Hohn-Schulte et al. 2013, Table 1). The mass coefficient C is measured on 157 extant fishes of known mass, width and depth (Brofeldt 1917, the "fishcatch" data set). It is tested by leave-one-species-out prediction, with a median absolute error of 4–14% across the five species used and 4–31% across all seven. None of these fishes is flattened like *Tiktaalik*, so C is applied outside the tested range of shapes.
- **Method A (reference):** FishBase Bayesian length–weight relationships (W = aL^b) for *Neoceratodus*, *Protopterus*, *Latimeria* and *Atractosteus*, with correlated uncertainty in (log a, b). This answers "what would an ordinary fish of this length weigh?"
- If the factors of Method B are independent, as the primary analysis assumes, Var(ln M) splits exactly into one term per factor (first-order Sobol indices).

## Main results (Method B, median and 95% interval)

| Specimen | Published reconstruction | Revised trunk (31 presacral vertebrae) |
|---|---|---|
| NUFV 108 (holotype, smallest) | 5.5 kg (3.0–10.4) | 4.3 kg (2.1–8.9) |
| NUFV 110 | 18.3 kg (7.8–43.3) | 14.3 kg (6.1–33.9) |
| NUFV 109 (largest articulated) | 43.9 kg (18.7–104.5) | 34.4 kg (14.7–81.3) |
| 2.75 m, read as excluding the caudal fin | 223 kg (79–628) | — |
| 2.75 m, read as total length | 139 kg (49–395) | — |

These intervals exclude model-form uncertainty. A rough across-species trend in C suggests the masses could be up to about 1.6–2.0 times higher, because *Tiktaalik* is flatter than every calibration fish.

The full output is in [`results/report.txt`](results/report.txt) and [`results/estimates.json`](results/estimates.json).

## Reproduce

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
python model.py > results/report.txt
```

This rewrites everything in `results/` and `figures/`. The random seed is fixed (20261102, N = 200,000 draws), so the numbers come out identical on every run.

## Files

| Path | Contents |
|---|---|
| `model.py` | The whole analysis: data loading, calibration, validation, both methods, sensitivity tests, figures |
| `data/fishcatch.dat` | Brofeldt (1917) measurements of 159 fishes from Lake Längelmävesi, Finland |
| `data/fishcatch_readme.txt` | The original description of the data set |
| `results/report.txt` | Human-readable summary of every number in the paper |
| `results/estimates.json` | Machine-readable results |
| `figures/` | Figures 1–5 of the paper |

## Data source

The fishcatch data come from Brofeldt, P. (1917), in Järvi, T. H., *Finlands Fiskerier* Band 4, Helsingfors. They are distributed by the *Journal of Statistics Education* data archive (contributed by J. Puranen). The 1917 measurements are in the public domain.

## Use of AI tools

During the preparation of this work, the author used Claude (Anthropic) as AI language model assistants. This tool was used to assist with the development and structuring of the custom Python scripts including code logic, interface design, and debugging, and to assist with the refinement of manuscript language for clarity and readability. All code and text outputs were reviewed, edited, and validated by the author, who takes full responsibility for the scientific content and interpretation presented in this work.

## Licence

Code: MIT ([`LICENSE`](LICENSE)). Results and figures: CC BY 4.0.

## Citation

If you use this code, please cite the paper above. GitHub's "Cite this repository" button reads [`CITATION.cff`](CITATION.cff).
