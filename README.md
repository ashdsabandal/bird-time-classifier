# When Can You See It? Predicting Bird Sighting Times with k-NN, Naive Bayes and CART

CMSC 170 project. Given a bird species, location, date and observation duration, we predict
which part of the day it is likely to be seen (**Early Morning, Late Morning, Afternoon or Night**),
and compare three classifiers implemented from scratch in Python.

- **Project page:** https://USERNAME.github.io/REPO-NAME/ <!-- EDIT -->
- **Slides:** SLIDES_URL <!-- EDIT -->
- **Task log:** [`TASK_LOG.md`](TASK_LOG.md)

## Group members

| Name | Main responsibilities |
|---|---|
| Member Name | |
| Member Name | |
| Member Name | |
| Member Name | |

## Repository structure

```
.
├── main.py               # Demo: runs each model on one test sighting
├── test_accuracy.py      # Evaluates all three models on the test set
├── src/
│   ├── __init__.py
│   ├── utils.py          # CSV loading, labeling, grouped train/test split, min-max scaling
│   ├── knn.py            # k-Nearest Neighbors
│   ├── naive_bayes.py    # Naive Bayes with binning and Laplace smoothing
│   ├── decision_tree.py  # CART decision tree (Gini impurity)
│   └── metrics.py        # Evaluation metrics
├── data/
│   └── dataset.csv       # eBird sightings (6,000 rows)
├── docs/                 # Project page served by GitHub Pages
├── TASK_LOG.md           # Member contributions
└── README.md
```

## Requirements

- Python 3.8 or newer (tested on Python 3.13)
- No third-party packages. Everything uses the standard library.
<!-- EDIT: if your plotting script uses matplotlib etc., list it here and add a requirements.txt -->

## Running the code

Run every command from the **repository root**, because the scripts load `data/dataset.csv`
by relative path.

```bash
git clone https://github.com/USERNAME/REPO-NAME.git
cd REPO-NAME
```

**1. Demo on a single sighting**

```bash
python main.py
```

This loads the data, trains all three models and classifies the first test sighting. It prints
the k-NN neighbors, the Naive Bayes class probabilities and the decision tree prediction.
It asks for the Naive Bayes smoothing value `k`; enter a positive integer such as `1`.

**2. Full evaluation**

```bash
python test_accuracy.py
```

This prints a side-by-side table of predictions for the first 20 test sightings, then the
accuracy and prediction counts of every model on the full test set. It takes about
10 seconds; most of that is k-NN.

## Reproducing our results

The results are deterministic. Nothing needs to be set by hand beyond the defaults:

| Setting | Value | Where |
|---|---|---|
| Shuffle seed | `67` | `src/utils.py`, `readFile` |
| Training size | ~4,800 rows (4,801 train / 1,199 test) | `split=4800` in `main.py` / `test_accuracy.py` |
| k-NN | `k = 10`, `p = 2` | `test_accuracy.py` |
| Naive Bayes | Laplace `k = 1`, 4 bins per numeric feature | `test_accuracy.py`, `src/naive_bayes.py` |
| Decision tree | `max_depth = 5`, `min_size = 10` | `test_accuracy.py` |

Expected output of `python test_accuracy.py` on the full test set:

```
k-NN Accuracy:          536/1199 (44.70%)
Naive Bayes Accuracy:   565/1199 (47.12%)
Decision Tree Accuracy: 564/1199 (47.04%)
```
<!-- EDIT: update these lines if the code or hyperparameters change, and add the other metrics once metrics.py is done -->

**Figures and tables on the project page:**
<!-- EDIT: describe the script/notebook that makes the graphs, e.g. -->
`python make_figures.py` *(to be added)* writes every chart to `docs/assets/figures/`.

## Dataset

`data/dataset.csv` (≈750 KB) is included in this repository.

- **Source:** [eBird](https://ebird.org), Cornell Lab of Ornithology
- **Region:** Laguna, Philippines (mostly Los Baños, Mt. Makiling, San Pedro, San Pablo City, Santa Rosa)
- **Dates:** January 2023 – August 2026
- **Size:** 6,000 sightings of 241 species
- **Columns:** `COMMON NAME, SCIENTIFIC NAME, OBSERVATION COUNT, LOCALITY, LATITUDE, LONGITUDE, OBSERVATION DATE, TIME OBSERVATIONS STARTED, DURATION MINUTES, EFFORT DISTANCE KM`

**To download it again from eBird:**
<!-- EDIT: write the exact steps you followed, e.g. -->
1. Create a free eBird account and request access to the data at https://ebird.org/data/download.
2. *(Filters used: region, date range, and anything else.)*
3. *(How the export was trimmed to these 6,000 rows and columns.)*
4. Save the result as `data/dataset.csv`.

The models use species, latitude, longitude, observation date, start time and duration.
The start time becomes the target label:

| Class | Start time |
|---|---|
| Early Morning | 04:00 – 06:59 |
| Late Morning | 07:00 – 11:59 |
| Afternoon | 12:00 – 16:59 |
| Night | 17:00 – 03:59 |

## Publishing the project page

The page lives in `docs/`. In the repository go to **Settings → Pages**, set *Source* to
**Deploy from a branch**, then choose branch `main` and folder **`/docs`**. The site appears at
`https://USERNAME.github.io/REPO-NAME/` within a minute or two. See `docs/README.md` for how to
fill in the page and add figures.

## References and disclosures

1. eBird. *eBird: An online database of bird distribution and abundance.* Cornell Lab of Ornithology, Ithaca, NY. https://ebird.org
2. J. Brownlee, *How To Implement The Decision Tree Algorithm From Scratch In Python*, Machine Learning Mastery. Basis for `src/decision_tree.py`.
3. *Scaling Numerical Data, Explained*, Towards Data Science. Used for min-max scaling in `src/utils.py`.
4. **AI tool use:** Claude (Anthropic) was used to generate the project page template in `docs/`, this README and the `TASK_LOG.md` skeleton. <!-- EDIT: add any other AI tools used and what for -->
