# PSC-PPGR code, figures, and tables

This repository contains the public code, final figures, final tables, and documentation files for the manuscript:

**PSC-PPGR: Personalized Support Calibration for Robust Postprandial Glycemic Response Mining under Device, Subject, and Setting Shifts**

To avoid redistributing participant-level processed biomedical data, this public GitHub version does **not** include processed event tables, train/test split files, subject-level prediction files, or external validation extracts. Those files should be deposited separately only after confirming the applicable data-use terms, or made available from the authors upon reasonable request.

## Contents

- `src/meter_ppgr/`: Core event construction, split, model, and evaluation code.
- `scripts/`: Analysis, table-generation, manuscript-generation, and figure-generation scripts.
- `scripts/figures/`: Matplotlib scripts for the main and supplementary figures.
- `scripts/tables/`: Scripts used to generate manuscript tables.
- `figures/final/`: Final main and supplementary figures in PNG/PDF/SVG formats, plus aggregate figure source CSV files where available.
- `tables/final/`: Final manuscript tables in Word format.
- `supplement/`: Supplementary material files and supplementary figures.
- `config.yaml`: Example configuration template.
- `requirements.txt`: Python package requirements.
-Currently, only the script file and supporting files have been uploaded. All files in the directory will be added after the paper is accepted.
## Data availability

Raw third-party datasets are not redistributed in this repository. Users should obtain the original CGMacros and AZT1D data from their official sources and comply with the corresponding data-use agreements.

Participant-level processed event tables, split files, and prediction-level outputs are intentionally excluded from this public GitHub repository. They can be deposited in a controlled or archival repository after author confirmation that redistribution is permitted.

## Environment

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Reproducing figures and tables

The figure scripts are designed to run independently when the expected aggregate source files are available.

```bash
python scripts/figures/fig5_main_performance.py
python scripts/figures/fig6_psc_robustness_bias.py
python scripts/tables/generate_main_tables_docx.py
```

Some full analysis scripts require processed event tables and split files that are not included in this public repository.
