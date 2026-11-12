import io
import contextlib
from collections import Counter
from src import utils, knn, naive_bayes, decision_tree

filename = "data/dataset.csv"
bird_rows = utils.readFile(filename)
training_data, test_data = utils.splitter(bird_rows, split=1400)

# 1. k-NN setup
with contextlib.redirect_stdout(io.StringIO()):
    knn_train, knn_test = utils.scale_features(
        knn.month_to_float(training_data),
        knn.month_to_float(test_data)
    )
k = 5
p = 2

# 2. Naive Bayes setup
ranges = naive_bayes.get_ranges(training_data)
nb_train = naive_bayes.make_features(training_data, ranges)
nb_test = naive_bayes.make_features(test_data, ranges)
bob, fcount, voc_count = naive_bayes.generate_bob(nb_train)
k_val = 1

# 3. Decision Tree (CART) setup
dt_train = decision_tree.format_rows(training_data)
dt_test = decision_tree.format_rows(test_data)
tree = decision_tree.build_tree(dt_train, max_depth=5, min_size=10)


def evaluate(n_samples, title, show_table=False):
    knn_correct = 0
    nb_correct = 0
    dt_correct = 0

    knn_preds = Counter()
    nb_preds = Counter()
    dt_preds = Counter()
    actuals = Counter()

    if show_table:
        print(f"\n=== {title} ===")
        print(f"{'#':<3} | {'Actual':<14} | {'k-NN Pred':<14} | {'NB Pred':<14} | {'CART Pred':<14}")
        print("-" * 70)

    for i in range(n_samples):
        actual = dt_test[i][-1]
        actuals[actual] += 1

        with contextlib.redirect_stdout(io.StringIO()):
            knn_pred = knn.predict(k, p, knn_train, knn_test[i])
        nb_pred, _ = naive_bayes.predict(k_val, bob, fcount, voc_count, nb_test[i])
        dt_pred = decision_tree.predict(tree, dt_test[i])

        knn_preds[knn_pred] += 1
        nb_preds[nb_pred] += 1
        dt_preds[dt_pred] += 1

        if knn_pred == actual:
            knn_correct += 1
        if nb_pred == actual:
            nb_correct += 1
        if dt_pred == actual:
            dt_correct += 1

        if show_table:
            print(f"{i+1:<3} | {actual:<14} | {knn_pred:<14} | {nb_pred:<14} | {dt_pred:<14}")

    if not show_table:
        print(f"\n=== {title} ===")
    else:
        print("-" * 70)

    print(f"Actual Counts:          {dict(actuals)}")
    print(f"k-NN Accuracy:          {knn_correct}/{n_samples} ({knn_correct / n_samples * 100:.2f}%) | Preds: {dict(knn_preds)}")
    print(f"Naive Bayes Accuracy:   {nb_correct}/{n_samples} ({nb_correct / n_samples * 100:.2f}%) | Preds: {dict(nb_preds)}")
    print(f"Decision Tree Accuracy: {dt_correct}/{n_samples} ({dt_correct / n_samples * 100:.2f}%) | Preds: {dict(dt_preds)}")


if __name__ == "__main__":
    evaluate(20, "First 20 Test Samples", show_table=True)
    evaluate(len(test_data), f"Full Test Set ({len(test_data)} Samples)", show_table=False)
