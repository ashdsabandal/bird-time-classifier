# Bird Watch: When Can You See 'Em? — Predicting Bird Sighting Times with k-NN, Naive Bayes and CART

CMSC 170 project. Given a bird species, location, date and observation duration, we predict
which part of the day it is likely to be seen (**Early Morning, Late Morning, Afternoon or Night**),
and compare three classifiers implemented from scratch in Python.

- **Project page:** https://ashdsabandal.github.io/bird-time-classifier/
- **Dataset:** [`data/dataset.csv`](data/dataset.csv)
- **Task log:** [`TASK_LOG.md`](TASK_LOG.md)

## Group members

| Name | Main responsibilities |
|---|---|
| Ashley Mark Sabandal | Project structure, initial eBird dataset (`data/dataset.csv`), Decision Tree (CART) implementation (`src/decision_tree.py`), accuracy evaluation script (`test_accuracy.py`), feature refinements in k-NN & Naive Bayes, Methods section on the project page (`docs/index.html`) |
| Joella Elindo | Evaluation metrics (`src/metrics.py`), full evaluation and method comparison script (`test_metrics.py`), Experimental Setup, result tables, learned rules, and visualizations (`docs/assets/figures/`, Setup & Results sections in `docs/index.html`) |
| Kristina Doroja | Dataset cleaning and expansion to 6,000 rows (`data/dataset.csv`), data loading, labeling, grouped train/test split and min-max scaling (`src/utils.py`), k-NN implementation (`src/knn.py`), Naive Bayes implementation (`src/naive_bayes.py`), single-sighting demo (`main.py`), project page & documentation including Problem, Dataset, Discussion, Conclusion, and References (`docs/`, `README.md`, `TASK_LOG.md`) |

## Repository structure

```
.
├── main.py               # Demo: trains all three models and classifies a single test sighting
├── test_accuracy.py      # Quick evaluation: prints first 20 predictions and overall accuracy
├── test_metrics.py       # Full evaluation: prints accuracy, precision, recall, F1, confusion matrices, and runtimes
├── src/
│   ├── __init__.py
│   ├── utils.py          # CSV loading, time-of-day labeling, grouped train/test split, min-max scaling
│   ├── knn.py            # k-Nearest Neighbors (Minkowski distance with circular date wrapping)
│   ├── naive_bayes.py    # Naive Bayes with quartile/seasonal binning and Laplace smoothing
│   ├── decision_tree.py  # CART decision tree (Gini impurity, binary splits)
│   └── metrics.py        # Accuracy, precision, recall, macro/weighted F1, confusion matrix, baseline
├── data/
│   └── dataset.csv       # Cleaned eBird sightings from Laguna, Philippines (6,000 rows)
├── docs/                 # Project page served by GitHub Pages (index.html, CSS, JS, figures)
├── TASK_LOG.md           # Log of member contributions
└── README.md
└── promptused.txt        # prompt used for webpage template generation

```

## Requirements

- **Python 3.8 or newer** (tested on Python 3.13)
- **No third-party packages required** to run the classifiers or evaluation scripts, everything is implemented using the Python standard library (`csv`, `datetime`, `random`, `time`, `collections`, `io`, `contextlib`).

## Where to locate the dataset

The dataset is included directly in the repository and mirrored online:

1. **In this repository:** [`data/dataset.csv`](data/dataset.csv)
2. **Online mirror (Google Sheets):** [eBird Laguna Dataset Sheet](https://docs.google.com/spreadsheets/d/10cOxFTLiOMKuFIsi7Tma6NHbXo2MzewQy_QRLe6tGng/edit?usp=sharing)
3. **Original source:** Obtained from [eBird](https://ebird.org) (Cornell Lab of Ornithology) by creating an eBird account and requesting observations filtered by **Region: Laguna, Philippines** (primarily Los Baños, Mt. Makiling, San Pedro, San Pablo City, and Santa Rosa) and **Date: January 2023 to August/October 2026**.

### Dataset summary & target classes

- **Size:** 6,000 complete sightings across 241 bird species
- **Input features (6):**
  - `Species` (`COMMON NAME`, categorical)
  - `Latitude` (`LATITUDE`, numeric)
  - `Longitude` (`LONGITUDE`, numeric)
  - `Month` & `Day` (parsed from `OBSERVATION DATE`, numeric)
  - `Duration` (`DURATION MINUTES`, numeric)
- **Target class (`Time of day`):** Derived from the hour of `TIME OBSERVATIONS STARTED`:

| Class | Start time | Count | Share |
|---|---|---:|---:|
| **Early Morning** | 04:00 – 06:59 | 3,017 | 50.3% |
| **Late Morning** | 07:00 – 11:59 | 1,268 | 21.1% |
| **Afternoon** | 12:00 – 16:59 | 1,368 | 22.8% |
| **Night** | 17:00 – 03:59 | 347 | 5.8% |

## Running the code

Run all commands from the **repository root** so the scripts can locate `data/dataset.csv` via relative path:

```bash
git clone https://github.com/ashdsabandal/bird-time-classifier.git
cd bird-time-classifier
```

### 1. Single-sighting demo (`main.py`)

```bash
python main.py
```

Loads `data/dataset.csv`, performs the grouped train/test split, trains all three classifiers, and predicts the time of day for the first test sighting:
- **k-NN:** Prints the `k = 5` closest neighbors (`p = 2`) with distances and labels.
- **Naive Bayes:** Prints class counts, vocabulary sizes, and class priors, then prompts you to enter a positive integer for the Laplace smoothing constant `k` (e.g., `1`) before printing posterior probabilities across the four time classes.
- **Decision Tree (CART):** Builds a tree (`max_depth = 5`, `min_size = 10`) and prints the predicted class.

### 2. Full metrics and method comparison (`test_metrics.py`)

```bash
python test_metrics.py
```

Runs the complete evaluation suite used for the project page (`docs/index.html`) using `src/metrics.py`. It outputs:
- Overall performance table (**Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1**) for **k-NN**, **Naive Bayes**, **CART**, and the **Majority-Class Baseline** (always predicting *Early Morning*).
- Runtime comparison (setup time, prediction time, and total runtime).
- Per-model classification reports (per-class precision, recall, F1-score, and support) and annotated 4×4 confusion matrices.
- Method comparison summary showing the best-performing model per metric and overall ranking by Macro F1.

## Reproducing our results

All experiments are deterministic and require no manual configuration:

| Setting | Value | Implementation details |
|---|---|---|
| **Random seed** | `67` | Fixed in `utils.readFile()` before shuffling rows |
| **Grouped train/test split** | `4,801` train (80%) / `1,199` test (20%) | `utils.splitter()` groups sightings by `(latitude, longitude, month, day, duration)` so checklists from the same outing never span both train and test sets (`split = 4800`) |
| **k-NN** | `k = 10`, `p = 2` | Coordinates and duration min-max scaled to `[0, 1]`; date converted to day-of-year fraction (`tm_yday / 366`) with circular wrapping `min(\|x-y\|, 1-\|x-y\|)` in `test_accuracy.py` (`skip=[3]`) and standard min-max scaling in `test_metrics.py` |
| **Naive Bayes** | Laplace `k = 1`, `4` bins | Species kept categorical; latitude, longitude, and duration discretized into 4 equal-frequency quartiles from training cutoffs; date binned into 4 equal seasons |
| **Decision Tree (CART)** | `max_depth = 5`, `min_size = 10` | Binary splits selected via weighted Gini impurity (`==` for categorical species, `<` for numeric latitude, longitude, month, day, duration) |

### Expected results on the held-out test set (1,199 sightings)

Running `python test_metrics.py` reproduces the exact numbers reported in Section 05 (*Results*) of `docs/index.html`:

| Method | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| **Baseline (always Early Morning)** | 43.12% | 10.78% | 25.00% | 15.06% | 25.98% |
| **k-NN** (`test_metrics.py`) | 44.62% | 34.92% | 31.72% | 30.50% | 40.15% |
| **Naive Bayes** | **47.12%** | **36.84%** | 34.88% | **34.39%** | 43.52% |
| **Decision Tree (CART)** | 47.04% | 33.17% | **35.03%** | 32.36% | **44.24%** |

