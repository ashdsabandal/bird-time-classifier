# Contains the implementation of the Decision Tree (CART) classification algorithm.
# Reference: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/


# Formats bird rows into [species, latitude, longitude, month, day, duration, classification]
def format_rows(bird_rows):
    formatted = []
    for r in bird_rows:
        formatted.append([
            r["species"],
            r["latitude"],
            r["longitude"],
            float(r["month"]),
            float(r["day"]),
            r["duration"],
            r["classification"]
        ])
    return formatted


# Splits dataset into left and right groups based on a feature and threshold/category
def test_split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        val = row[index]
        if isinstance(value, str):
            if val == value:
                left.append(row)
            else:
                right.append(row)
        else:
            if val < value:
                left.append(row)
            else:
                right.append(row)
    return left, right


# Calculates weighted Gini impurity for split groups
def gini_index(groups, classes):
    n_instances = float(sum(len(group) for group in groups))
    if n_instances == 0:
        return 0.0

    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        labels = [row[-1] for row in group]
        for class_val in classes:
            p = labels.count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)

    return gini


# Evaluates candidate splits across all features to find the lowest Gini score
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999.0, None
    n_features = len(dataset[0]) - 1

    for index in range(n_features):
        unique_values = set(row[index] for row in dataset)
        for value in unique_values:
            groups = test_split(index, value, dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini, groups

    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Selects the most common class label in a group to form a leaf node
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Recursively creates child splits or terminal leaf nodes
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del node['groups']

    # No-split condition: all samples went one side
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # Max depth reached
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # Process left child
    if len(left) <= min_size or len(set(row[-1] for row in left)) == 1:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    # Process right child
    if len(right) <= min_size or len(set(row[-1] for row in right)) == 1:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# Builds a CART decision tree from training data
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# Traverses tree with a given row until reaching a terminal leaf label
def predict(node, row):
    val = row[node['index']]
    if isinstance(node['value'], str):
        is_left = (val == node['value'])
    else:
        is_left = (val < node['value'])

    branch = node['left'] if is_left else node['right']
    if isinstance(branch, dict):
        return predict(branch, row)
    return branch


# Predicts classifications for a batch of rows
def batch_predict(tree, rows):
    return [predict(tree, row) for row in rows]


# Prints human-readable if/else decision rules from the tree
def print_tree(node, depth=0, feature_names=None):
    if isinstance(node, dict):
        feat = feature_names[node['index']] if feature_names else f"X{node['index']}"
        op = "==" if isinstance(node['value'], str) else "<"
        print(f"{'  ' * depth}[{feat} {op} {node['value']}]")
        print(f"{'  ' * depth}--> True:")
        print_tree(node['left'], depth + 1, feature_names)
        print(f"{'  ' * depth}--> False:")
        print_tree(node['right'], depth + 1, feature_names)
    else:
        print(f"{'  ' * depth}Predict: {node}")
