# Contains evaluation metric functions implemented to evaluate model performance.

# References:
# Sokolova & Lapalme (2009)
# Opitz (2024)

LABELS = ["Early Morning", "Late Morning", "Afternoon", "Night"]


# counts matching actual and predicted divided by total 
# so number of tama
def accuracy(actual, predicted):
    correct = 0

    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / len(actual)

# mas intuitive sya sa test_metrics pero cinocompare lang yung actual/predicted side by side
def confusion_matrix(actual, predicted):
    n = len(LABELS)
    matrix = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(len(actual)):
        row = LABELS.index(actual[i])
        col = LABELS.index(predicted[i])

        matrix[row][col] += 1

    return matrix

# includes precision, recall, f1 score, and support
def classification_report(actual, predicted):
    matrix = confusion_matrix(actual, predicted)
    report = {}

    for i in range(len(LABELS)):
        tp = matrix[i][i]
        fp = sum(row[i] for row in matrix) - tp
        fn = sum(matrix[i]) - tp

        support = sum(matrix[i])

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0
        )

        report[LABELS[i]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }

    return report

# including other evaluation metrics from classification metrics research
def evaluate(actual, predicted):
    report = classification_report(actual, predicted)

    n = len(LABELS)
    total = len(actual)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    weighted_f1 = 0

    for label in LABELS:
        macro_precision += report[label]["precision"]
        macro_recall += report[label]["recall"]
        macro_f1 += report[label]["f1"]

        weighted_f1 += (
            report[label]["f1"] * report[label]["support"]
        )

    return {
        "accuracy": accuracy(actual, predicted),
        "precision": macro_precision / n,
        "recall": macro_recall / n,
        "f1": macro_f1 / n,
        "weighted_f1": weighted_f1 / total
    }


# Majority-class baseline
def baseline(training_labels, actual):
    counts = {}

    for label in training_labels:
        counts[label] = counts.get(label, 0) + 1

    majority = max(counts, key=counts.get)

    predicted = [majority for _ in actual]

    return predicted
