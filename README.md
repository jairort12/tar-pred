\# TAR Prediction – Inference \& Model Evaluation



This repository ships \*\*pre-trained ML/DL models\*\* and the required preprocessing artifacts to:

1\) \*\*Predict TAR (g/Nm³)\*\* from a raw CSV following a defined template.

2\) \*\*Evaluate all models\*\* on a saved test split and generate \*\*R²\*\* metrics and \*\*y\_true vs y\_pred\*\* plots.



> This repo is focused on \*\*inference and evaluation\*\*. The full training workflow and methodology are documented in the associated paper / supplementary material.



---



\## Repository structure



\- `models/`  

&nbsp; Pre-trained model pipelines (`\*.pkl`). Includes an optional stacking model.

\- `artifacts/`  

&nbsp; Clustering artifacts (scalers + kmeans) used to compute `BM\_Cluster\_Final`.

\- `docs/`  

&nbsp; CSV templates and data dictionary (units, categorical values, constraints).

\- `src/tar\_pred/`  

&nbsp; Python package for preprocessing, prediction, and evaluation.

\- `data/processed/`  

&nbsp; Expected local location for `X\_test.pkl` and `y\_test.pkl` when running `evaluate`.

\- `reports/`  

&nbsp; Output folder (ignored by git): metrics, figures, prediction reports.



---



\## Data input: CSV template and units



The raw input must use \*\*semicolon (`;`)\*\* as separator and must follow the templates:



\- `docs/template\_raw.csv` (prediction – no TAR column)

\- `docs/template\_eval.csv` (evaluation – includes TAR column)



\### Units and bases (required)

\- Proximate analysis (`FC`, `VM`, `Ash`): \*\*dry basis\*\*, % mass

\- Ultimate analysis (`pC`, `pH`, `pO`, `pN`, `pS`): \*\*dry basis\*\*, % mass

\- Gas composition (`H2`, `CH4`, `CO`, `CO2`, `N2`): \*\*% v/v\*\*

\- `ER`, `SyBR`: dimensionless

\- `Tred`: °C

\- `SF`: Nm³/h

\- `BF`: kg/h

\- `TAR`: g/Nm³ (only required for evaluation datasets)



\### Allowed categorical values (template)

\- `GT`: `BFB`, `DG`, `FB`, `CFB`, `DFB`, `UG`

\- `GA`: `Air`, `Air-Steam`, `Steam`, `O2-steam`, `O2`

\- `MM`: `TP`, `SPA`, `GC`, `GC-MS`



\### Training-consistent exclusions (warning + drop rows)

If present, the following rows are \*\*excluded\*\* (models were never trained with them):

\- `GT ∈ {DTBG, Entrained flow}`

\- `GA = CO2`



The code will emit warnings and drop those rows (it will not hard-fail).



---



\## Preprocessing performed during inference



The inference pipeline reproduces the training preparation order:



1\) Drop column `ID` (kept only for output reference if present).

2\) Gas normalization \*\*free of N2\*\* (H2/CO/CH4/CO2 rescaled to sum to 100).

3\) Proximate normalization (FC/VM/Ash rescaled to sum to 100).

4\) Drop column `N2`.

5\) Compute `BM\_Cluster\_Final` via clustering + subclustering (from `artifacts/`).

6\) Drop training-removed variables when present: `\['Ash', 'VM', 'pO', 'BF', 'BM']`.

7\) Feature engineering via `FeatureBuilder` (embedded in model pipelines).

8\) Final encoding/scaling is performed inside each saved model pipeline.



---



\## Installation



\### Recommended: create a virtual environment (Windows / Git Bash)



From the repository root:



```bash

py -3.12 -m venv tar-pred-venv

source tar-pred-venv/Scripts/activate

python -m pip install -U pip

python -m pip install -e . --no-deps



Dependencies and reproducibility



This repository includes:



requirements.lock.txt: an environment snapshot for reference.



Important: Loading pickled scikit-learn pipelines is most reliable when using compatible

versions of scikit-learn (and sometimes numpy). If you see InconsistentVersionWarning,

consider aligning versions to those used during training.



A minimal dependency set typically includes:



numpy, pandas, scipy



scikit-learn



joblib



matplotlib



optional (required by some models): xgboost, lightgbm, catboost



Tested environment: \[fill this with your final local versions]

Example: Python 3.12.x, scikit-learn 1.6.1, numpy 2.x



Usage

1\) Predict from a raw CSV

tar-predict predict \\

&nbsp; --input data/raw/your\_data.csv \\

&nbsp; --model-path models/your\_model.pkl \\

&nbsp; --artifacts-dir artifacts \\

&nbsp; --out reports/predictions.csv \\

&nbsp; --report reports/prediction\_report.json



Output:



reports/predictions.csv (contains ID if provided + y\_pred)



reports/prediction\_report.json (warnings, exclusions, pipeline metadata)



2\) Evaluate all models on X\_test/y\_test



Place these files locally:



data/processed/X\_test.pkl



data/processed/y\_test.pkl



Then run:



tar-predict evaluate \\

&nbsp; --models-dir models \\

&nbsp; --x-test data/processed/X\_test.pkl \\

&nbsp; --y-test data/processed/y\_test.pkl \\

&nbsp; --outdir reports



Output:



reports/metrics.csv (R² per model, sorted best-to-worst)



reports/figures/\*.png (scatter plots with diagonal y=x)



reports/fallos.csv (only if a model fails to load/run)



Notes on model loading and security



This project uses joblib.load() to load model pipelines (.pkl). Pickle files can execute

arbitrary code if tampered with. Only load model files you trust.



Citation



If you use this repository in academic work, please cite the associated paper:



\[Add your paper citation here]



License



\[Add your chosen license here]

