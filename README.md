# DS203 Project - Nation Clustering

Clustering countries by socioeconomic indicators (World Bank WDI) and comparing the clusters with
World Bank income groups and geographical regions.

**Pipeline:** data cleaning -> standardisation -> correlation analysis -> PCA -> clustering -> interpretation

## Setup (Windows, PowerShell, from the project folder)

```powershell
cd "C:\IITB 7th sem\DS203\Project"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

In VS Code: `Ctrl+Shift+P` -> *Python: Select Interpreter* -> pick `.venv`. Open a notebook and choose the same
`.venv` as the kernel (top-right).

## Data

1. Download the **WDI CSV bulk download** from <https://datatopics.worldbank.org/world-development-indicators/>
   (Data -> Bulk download -> CSV).
2. Unzip it and put `WDICSV.csv` (or the whole unzipped folder) into `data/raw/`. It is git-ignored (too large).
3. Country metadata (region, income group) is fetched from the World Bank API by notebook 01 and cached in
   `data/raw/wb_country_api.json`.

Data downloaded on: **<fill in date>**   |   Year used: **<fill in after notebook 01>**

## Notebooks (run in order)

| Notebook | Content | Status |
|---|---|---|
| `01_data_building.ipynb` | metadata, aggregates removed, year choice, missingness analysis, dropping rule, KNN imputation | ready |
| `02_eda.ipynb` | summary stats, distributions, transformations, income-group comparison, correlations, scatter plots | ready |
| `03_pca.ipynb` | standardisation, PCA, explained variance, loadings | todo |
| `04_clustering.ipynb` | K-means, hierarchical clustering, comparison with income groups / regions, unusual countries | todo |

Shared constants and transformations live in `src/common.py` (edit transformations **there**, once).

## Folder layout

```
data/raw/         WDI zip/CSV + API cache (not committed)
data/processed/   cleaned tables produced by notebook 01 (committed)
notebooks/        01..04
reports/figures/  saved plots
src/common.py     indicator list, colours, transformations
```

## Decision log (fill in as you go; every choice needs a justification)

| Decision | Choice | Justification |
|---|---|---|
| Year *t* | | |
| Countries dropped (rule) | | |
| Imputation | | |
| Transformations | | |
| Number of PCs | | |
| Number of clusters *k* | | |
| Hierarchical linkage | | |

> Income group and region are **never** used as clustering features - only to interpret and evaluate clusters.
