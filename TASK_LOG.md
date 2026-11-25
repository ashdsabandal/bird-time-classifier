# Task Log

Record of who did what. Add a row whenever a task is finished. Newest entries go at the bottom.

| Date | Member | Task | Files | Status |
|---|---|---|---|---|
| 2026-10-29 | Ashley Mark Sabandal | Initial repository structure and initial 2,000-row eBird dataset | `main.py`, `src/`, `data/dataset.csv` | Done |
| 2026-11-03 | Kristina Doroja | Cleaned dataset for complete rows; implemented CSV loading, labeling, grouped train/test split, and min-max scaling | `data/dataset.csv`, `main.py`, `src/utils.py` | Done |
| 2026-11-03 | Kristina Doroja | Implemented k-Nearest Neighbors (k-NN) classifier | `src/knn.py`, `main.py` | Done |
| 2026-11-04 | Kristina Doroja | Implemented Naive Bayes classifier and documentation | `src/naive_bayes.py`, `main.py` | Done |
| 2026-11-04 | Ashley Mark Sabandal | Implemented Decision Tree (CART) classifier and integrated into `main.py` | `src/decision_tree.py`, `main.py` | Done |
| 2026-11-12 | Ashley Mark Sabandal | Included species in k-NN/NB feature extraction; split Morning into Early and Late Morning | `src/knn.py`, `src/naive_bayes.py`, `src/utils.py` | Done |
| 2026-11-13 | Ashley Mark Sabandal | Added test accuracy evaluation script | `test_accuracy.py` | Done |
| 2026-11-13 | Kristina Doroja | Expanded dataset to 6,000 rows; updated date/time evaluation and Naive Bayes feature binning | `data/dataset.csv`, `src/knn.py`, `src/naive_bayes.py`, `src/utils.py`, `test_accuracy.py` | Done |
| 2026-11-18 | Kristina Doroja | Created project page template, README, and task log; initial page edits | `docs/`, `README.md`, `TASK_LOG.md` | Done |
| 2026-11-18 | Joella Elindo | Implemented evaluation metrics and comprehensive method comparison script | `src/metrics.py`, `test_metrics.py` | Done |
| 2026-11-24 | Ashley Mark Sabandal | Wrote Methods section (k-NN, Naive Bayes, CART) on the project page | `docs/index.html` | Done |
| 2026-11-25 | Kristina Doroja | Populated Problem, Dataset, Limitations, Improvements, References, and links on the project page | `docs/index.html` | Done |
| 2026-11-25 | Joella Elindo | Generated all charts/figures and populated Results and Setup tables on the project page | `docs/assets/figures/`, `docs/index.html` | Done |
| 2026-11-25 | Joella Elindo | Experimental Setup: How hyperparameter values were chosen | `docs/index.html` | Done |
| 2026-11-25 | Joella Elindo | Results: Paste learned Decision Tree rules (`print_tree` output) | `docs/index.html` | Done |
| 2026-11-25 | Kristina Doroja | Discussion: Why each method performed as it did (k-NN, Naive Bayes, Decision Tree) | `docs/index.html` | Done |
| 2026-11-25 | Kristina Doroja | Discussion: Influence of evaluation metrics & remaining limitation/improvement bullets | `docs/index.html` | Done |
| 2026-11-25 | Kristina Doroja | Conclusion: Key findings 1–3 | `docs/index.html` | Done |

