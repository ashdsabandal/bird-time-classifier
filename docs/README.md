# Project page (`docs/`)

This folder is the GitHub Pages site. It's plain HTML/CSS/JS with no build step.

```
index.html              ← all page content
assets/css/style.css    ← colors and fonts (see :root at the top)
assets/js/main.js       ← figure loading, tabs, nav highlight
assets/figures/         ← chart images go here
```

## Filling in the page

- Search `index.html` for **`EDIT:`** to find every spot that needs content.
- Dashed boxes (`<span class="todo">…</span>`) are text still to be written. Replace the whole span.
- Table cells showing `—` have `class="tbd"`. Put the number in and remove `tbd`.
  Add `class="best"` to highlight the top value in a column.
- Replace **`REPO_URL`** and **`SLIDES_URL`** everywhere (hero buttons and section 09).

## Adding graphs

Each figure box already has a file name. Save the chart with that exact name in
`assets/figures/` and it appears automatically.

| File name | Section |
|---|---|
| `class_distribution.png` | 02 Dataset |
| `sightings_map.png` | 02 Dataset |
| `pipeline.png` | 03 Methods (optional) |
| `model_comparison.png` | 05 Results, overall |
| `per_class_f1.png` | 05 Results, per class |
| `prediction_distribution.png` | 05 Results, per class |
| `knn_confusion_matrix.png` | 05 Results › k-NN |
| `knn_k_vs_score.png` | 05 Results › k-NN |
| `knn_p_vs_score.png` | 05 Results › k-NN |
| `nb_confusion_matrix.png` | 05 Results › Naive Bayes |
| `nb_smoothing_vs_score.png` | 05 Results › Naive Bayes |
| `nb_class_probabilities.png` | 05 Results › Naive Bayes |
| `dt_confusion_matrix.png` | 05 Results › Decision Tree |
| `dt_depth_vs_score.png` | 05 Results › Decision Tree |
| `dt_tree.png` | 05 Results › Decision Tree |

From matplotlib: `plt.savefig("docs/assets/figures/knn_k_vs_score.png", dpi=200, bbox_inches="tight")`

For `.svg`, `.jpg` or an interactive Plotly `.html`, change that figure's `data-src` in
`index.html`. To add a figure, copy any `<figure>` block and give it a new file name.

## Previewing locally

Open `index.html` in a browser. Interactive `.html` figures need a local server:
`python -m http.server --directory docs`, then open `http://localhost:8000`.
