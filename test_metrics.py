# based lang din sa test_accuracy pero ginamit yung metrics.py

import io
import time
import contextlib
from collections import Counter
from src import utils, knn, naive_bayes, decision_tree, metrics


# Load and partition the dataset.
filename = "data/dataset.csv"
bird_rows = utils.readFile(filename)

training_data, test_data = utils.splitter(
    bird_rows, split=int(len(bird_rows) * 0.8)
)


# 1. k-NN setup
start = time.perf_counter()

with contextlib.redirect_stdout(io.StringIO()):
    knn_train, knn_test = utils.scale_features(
        knn.month_to_float(training_data),
        knn.month_to_float(test_data)
    )

k = 10
p = 2

knn_setup_time = time.perf_counter() - start


# 2. Naive Bayes setup
start = time.perf_counter()

ranges = naive_bayes.get_ranges(training_data)
nb_train = naive_bayes.make_features(training_data, ranges)
nb_test = naive_bayes.make_features(test_data, ranges)

bob, fcount, voc_count = naive_bayes.generate_bob(nb_train)
k_val = 1

nb_setup_time = time.perf_counter() - start


# 3. Decision Tree (CART) setup
start = time.perf_counter()

dt_train = decision_tree.format_rows(training_data)
dt_test = decision_tree.format_rows(test_data)

tree = decision_tree.build_tree(
    dt_train, max_depth=5, min_size=10
)

dt_setup_time = time.perf_counter() - start


# Evaluate the three models using the same test dataset.
def evaluate(n_samples, title, show_table=False):

    # Store actual classifications and model predictions.
    actuals = []
    knn_preds = []
    nb_preds = []
    dt_preds = []

    # Store prediction runtimes.
    knn_time = 0
    nb_time = 0
    dt_time = 0

    print(f"\n=== {title} ===")

    if show_table:
        print(f"{'#':<4} | {'Actual':<14} | {'k-NN':<14} | {'NB':<14} | {'CART':<14}")
        print("-" * 75)

    for i in range(n_samples):
        actual = dt_test[i][-1]

        # Generate k-NN prediction and measure runtime.
        start = time.perf_counter()

        with contextlib.redirect_stdout(io.StringIO()):
            knn_pred = knn.predict(
                k, p, knn_train, knn_test[i]
            )

        knn_time += time.perf_counter() - start

        # Generate Naive Bayes prediction and measure runtime.
        start = time.perf_counter()

        nb_pred, _ = naive_bayes.predict(
            k_val, bob, fcount, voc_count, nb_test[i]
        )

        nb_time += time.perf_counter() - start

        # Generate CART prediction and measure runtime.
        start = time.perf_counter()

        dt_pred = decision_tree.predict(tree, dt_test[i])

        dt_time += time.perf_counter() - start

        # Store actual and predicted classifications.
        actuals.append(actual)
        knn_preds.append(knn_pred)
        nb_preds.append(nb_pred)
        dt_preds.append(dt_pred)

        if show_table:
            print(f"{i+1:<4} | {actual:<14} | {knn_pred:<14} | {nb_pred:<14} | {dt_pred:<14}")

    # Generate majority-class baseline predictions.
    training_labels = [
        row["classification"] for row in training_data
    ]

    baseline_preds = metrics.baseline(
        training_labels, actuals
    )

    models = {
        "k-NN": knn_preds,
        "Naive Bayes": nb_preds,
        "CART": dt_preds,
        "Baseline": baseline_preds
    }

    runtimes = {
        "k-NN": knn_time,
        "Naive Bayes": nb_time,
        "CART": dt_time
    }

    setup_times = {
        "k-NN": knn_setup_time,
        "Naive Bayes": nb_setup_time,
        "CART": dt_setup_time
    }

    print("\nActual Counts:")
    print(dict(Counter(actuals)))

    # Display overall performance comparison.
    print("\n=== OVERALL PERFORMANCE ===")

    print(f"{'Model':<16} {'Accuracy':>10} {'Macro P':>10} {'Macro R':>10} {'Macro F1':>10} {'Weighted F1':>12}")
    print("-" * 75)

    for name, predictions in models.items():

        scores = metrics.evaluate(actuals, predictions)

        print(
            f"{name:<16} "
            f"{scores['accuracy']:>10.2%} "
            f"{scores['precision']:>10.2%} "
            f"{scores['recall']:>10.2%} "
            f"{scores['f1']:>10.2%} "
            f"{scores['weighted_f1']:>12.2%}"
        )

    # Compare computational efficiency.
    print("\n=== RUNTIME COMPARISON ===")

    print(f"{'Model':<16} {'Setup (s)':>12} {'Predict (s)':>12} {'Total (s)':>12}")

    for name in runtimes:
        setup = setup_times[name]
        prediction = runtimes[name]

        print(
            f"{name:<16} "
            f"{setup:>12.6f} "
            f"{prediction:>12.6f} "
            f"{setup + prediction:>12.6f}"
        )

    # Display detailed performance per model.
    for name, predictions in models.items():

        print(f"\n=== {name.upper()} ===")

        print("Prediction Counts:")
        print(dict(Counter(predictions)))

        report = metrics.classification_report(
            actuals, predictions
        )

        print("\nClassification Report:")

        print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'Support':>10}")
        print("-" * 60)

        for label, scores in report.items():
            print(
                f"{label:<16} "
                f"{scores['precision']:>10.2%} "
                f"{scores['recall']:>10.2%} "
                f"{scores['f1']:>10.2%} "
                f"{scores['support']:>10}"
            )

        # Display confusion matrix.
        matrix = metrics.confusion_matrix(actuals, predictions)

        labels = metrics.LABELS
        short_labels = ["Early AM", "Late AM", "Afternoon", "Night"]

        print("\n================ CONFUSION MATRIX ================")
        print("\nRows    : Actual time category")
        print("Columns : Predicted time category")
        print("(*)     : Correct classifications\n")

        # Display column headings.
        print(f"{'':<17}", end="")

        for label in short_labels:
            print(f"| {label:>10} ", end="")

        print(f"| {'Total':>7}")

        border = "-" * 78
        print(border)

        # Display classification counts per class.
        for i in range(len(labels)):
            print(f"{labels[i]:<17}", end="")

            for j in range(len(labels)):
                value = str(matrix[i][j])

                # Mark correct classifications with an asterisk.
                if i == j:
                    value += "*"

                print(f"| {value:>10} ", end="")

            print(f"| {sum(matrix[i]):>7}")

        print(border)

        # Display total predictions per class.
        print(f"{'Total':<17}", end="")

        for j in range(len(labels)):
            total = sum(row[j] for row in matrix)
            print(f"| {total:>10} ", end="")

        print(f"| {len(actuals):>7}")

        # Summarize correctly classified observations per class.
        print("\nCLASSIFICATION SUMMARY\n")

        for i in range(len(labels)):
            correct = matrix[i][i]
            total = sum(matrix[i])
            recall = correct / total if total > 0 else 0

            print(
                f"{labels[i]:<15}: "
                f"{correct:>4} out of {total:>4} "
                f"correctly classified ({recall:.2%})"
            )

        # Summarize overall classification performance.
        correct = sum(matrix[i][i] for i in range(len(labels)))
        total = len(actuals)

        print("\nOverall Results:")
        print(f"    Total observations  : {total}")
        print(f"    Correct predictions : {correct}")
        print(f"    Wrong predictions   : {total - correct}")
        print(f"    Overall accuracy    : {correct / total:.2%}")

    compare_methods(actuals, models, runtimes, setup_times)

# Compares the predictive performance and efficiency of all three methods.
def compare_methods(actuals, models, runtimes, setup_times):

    results = {}
    reports = {}

    # Compute overall and per-class metrics for each method.
    for name in models:
        results[name] = metrics.evaluate(actuals, models[name])

        reports[name] = metrics.classification_report(
            actuals, models[name]
        )

    methods = ["k-NN", "Naive Bayes", "CART"]

    # Display comparison table headings.
    print("\n================ METHOD COMPARISON ================\n")

    print(
        f"{'Metric':<22} | "
        f"{'k-NN':>12} | "
        f"{'Naive Bayes':>12} | "
        f"{'CART':>12} | "
        f"{'Best Method':>15}"
    )

    print("-" * 86)

    evaluation_metrics = {
        "Accuracy": "accuracy",
        "Macro Precision": "precision",
        "Macro Recall": "recall",
        "Macro F1-score": "f1",
        "Weighted F1-score": "weighted_f1"
    }

# I GENERATED A SUMMARY LANG
    # Compare overall classification performance.
    for label, key in evaluation_metrics.items():

        values = [results[m][key] for m in methods]

        best = max(values)
        winners = [
            m for m in methods if results[m][key] == best
        ]

        winner = winners[0] if len(winners) == 1 else "Tie"

        print(
            f"{label:<22} | "
            f"{values[0]:>12.2%} | "
            f"{values[1]:>12.2%} | "
            f"{values[2]:>12.2%} | "
            f"{winner:>15}"
        )

    print("-" * 86)

    # Compare F1-scores for each time category.
    for label in metrics.LABELS:

        values = [
            reports[m][label]["f1"] for m in methods
        ]

        best = max(values)
        winners = [
            m for m in methods
            if reports[m][label]["f1"] == best
        ]

        winner = winners[0] if len(winners) == 1 else "Tie"

        print(
            f'{label + " F1":<22} | '
            f"{values[0]:>12.2%} | "
            f"{values[1]:>12.2%} | "
            f"{values[2]:>12.2%} | "
            f"{winner:>15}"
        )

    print("-" * 86)

    # Organize runtime measurements.
    time_metrics = {
        "Setup Time": setup_times,
        "Prediction Time": runtimes,
        "Total Runtime": {
            m: setup_times[m] + runtimes[m]
            for m in methods
        }
    }

    # Compare computational efficiency.
    for label, times in time_metrics.items():

        values = [times[m] for m in methods]

        best = min(values)
        winners = [
            m for m in methods if times[m] == best
        ]

        winner = winners[0] if len(winners) == 1 else "Tie"

        print(
            f"{label:<22} | "
            f"{values[0]:>11.4f}s | "
            f"{values[1]:>11.4f}s | "
            f"{values[2]:>11.4f}s | "
            f"{winner:>15}"
        )

    print("-" * 86)

    # Rank methods according to macro F1-score.
    ranking = sorted(
        methods,
        key=lambda m: results[m]["f1"],
        reverse=True
    )

    print("\nOVERALL RANKING (MACRO F1-SCORE)")

    for i, name in enumerate(ranking):
        print(f"{i + 1}. {name}: {results[name]['f1']:.2%}")

# Run the evaluation on the entire test dataset.
if __name__ == "__main__":

    print("Dataset size:", len(bird_rows))
    print("Training samples:", len(training_data))
    print("Testing samples:", len(test_data))

    evaluate(
        len(test_data),
        "Full Test Set",
        show_table=False
    )
