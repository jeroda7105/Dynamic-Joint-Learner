

# Sensitivity Analysis to **F** (NN-GP layers)

This repository reproduces the **sensitivity analysis to the hyperparameter F** (the number of layers used to fit the model’s Neural Network Gaussian Processes) from the manuscript:

**“Advancing Counter-Terrorism: Predicting Hidden Alliances and Rivalries in Evolving Terrorism Graphs with Organizational Attributes”**

Running the workflow reproduces **Tables 8, 9, and 10** (including subtables) from the supplementary materials of the manuscript,
Section **2.2.2** (*Sensitivity Analysis → Sensitivity to F*).

---

## Requirements

- **R:** 4.5.1  
- **Package management:** this project is intended to be run with **`renv`** (package versions are pinned in `renv.lock`).

---

## Quick start (recommended)

### 1) Clone the repository

Using HTTPS:

```bash
git clone https://github.com/jeroda7105/Dynamic-Joint-Learner.git
cd Dynamic-Joint-Learner
```

Or using SSH:

```bash
git clone git@github.com:jeroda7105/Dynamic-Joint-Learner.git
cd Dynamic-Joint-Learner
```


### 2) Open the RStudio project (recommended) / set the working directory

**Recommended (RStudio):** open `Dynamic-Joint-Learner.Rproj`. This will automatically set the working directory to the **repository root**.

**Alternative (without opening the `.Rproj`):** ensure your working directory is the repository root (the folder containing `F_sensitivity_analysis.Rmd`, `data/`, and `code/`), e.g.

```r
setwd("/path/to/Dynamic-Joint-Learner")
```

(You can also set this in RStudio via _Session → Set Working Directory…_.)


### 3) Restore the reproducible R environment with `renv`

```r
install.packages("renv")   # if needed
renv::restore()
```

### 4) Run the analysis

You can knit in RStudio, or render from the console:

```r
rmarkdown::render("F_sensitivity_analysis.Rmd")
```

The rendered PDF output is:

- `F_sensitivity_analysis.pdf`

---
## Further Details

### Parallel computing

`F_sensitivity_analysis.Rmd` uses **`doParallel`** and **`foreach`** to run simulation cases and replications in parallel.

- The original runs used **6 cores**.
- You can change this using the variable **`n_workers`** in `F_sensitivity_analysis.Rmd`.

If you experience heavy CPU usage or instability, reduce `n_workers`.

---

### Inputs and outputs

### Inputs (simulated datasets)

- Simulated datasets used in the sensitivity analysis are stored as `.RData` files under `data/`. 
  The data dictionary for simulated datasets are in the file `data/simulated_data_dictionary.txt`.


### Outputs (intermediate + final)

During execution, the workflow will create a folder (if it does not already exist):

- `code/outputs/`

This folder stores `.RData` files containing **intermediate results** from different model runs.

At the end of the workflow, results are aggregated across replications and cases to produce the tables corresponding to the three sensitivity-analysis cases (Tables 8–10), including:

- AUC of edge prediction  
- MSPE of nodal attribute prediction  
- 95% prediction interval coverage for nodal attributes  
- 95% prediction interval length for nodal attributes  

---

### Key files / repository structure

### Main analysis

- `F_sensitivity_analysis.Rmd` — R Markdown source used to replicate the sensitivity analysis tables.
- `F_sensitivity_analysis.pdf` — rendered output of the R Markdown (included for reference).

### Core method implementation

- `code/incomplete_graphs_model.R` — implementation for the proposed method (DJL).  
  Fits the DJL model using the dynamic multiplex graph `C_arr` and nodal attributes `x_arr`, producing MCMC samples for model parameters.

### Supporting code (in `code/`)

- `code/model_development.R` — intermediate calculations used in data generation and in model fitting.
- `code/model_generation_running_and_testing_functions.R` — helper functions to fit DJL and obtain prediction / uncertainty quantification results.
- `code/F_sensitivity_results_function.R` — aggregates results across replications for a given sensitivity-analysis case.

### Simulated data generation scripts (Scenarios 1–3)

- `code/simulated_data_generation_nngp_layerwise_missingness.R` — Scenario 1 (**used for the sensitivity analysis here**)
- `code/simulated_data_generation_layerwise_missingness.R` — Scenario 2
- `code/simulated_data_generation_tergm_layerwise_missingness.R` — Scenario 3

---

### Notes

- If you see “file not found” errors, confirm your working directory is the **repo root**, `Dynamic-Joint-Learner`. 
- If you see package/version errors, rerun `renv::restore()` and confirm you are using **R 4.5.1**.


