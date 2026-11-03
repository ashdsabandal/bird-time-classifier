# Serves as the main entry point to load data, train the three models, evaluate their performance, and compare results.

from src import utils, knn, decision_tree

filename = "data/dataset.csv"

bird_rows = utils.readFile(filename)
training_data, test_data = utils.splitter(bird_rows, split = 1400)

# KNN 
  # FOR JOEY AND ASH: BAKA NEED NYO RIN GAWIN TONG STUFF NA TO KASI 
  # FOR FORMATTING SYA , ESPECIALLY SI SCALE FEATURES 
knn_train = knn.month_to_float(training_data)
knn_test = knn.month_to_float(test_data)

knn_train, knn_test = utils.scale_features(knn_train, knn_test)

# k = int(input("Enter K: "))
# p = float(input("Enter p: "))
k = 5
p = 2

bird = knn_test[0]
prediction = knn.predict(k,p,knn_train, bird)

print(f"\n Species: {test_data[0]['species']}")
print(f"Listed Classification: {bird[-1]}")
print(f"Calculated Classification: {prediction}")

# Decision Tree (CART)
dt_train = decision_tree.format_rows(training_data)
dt_test = decision_tree.format_rows(test_data)

tree = decision_tree.build_tree(dt_train, max_depth=5, min_size=10)
dt_bird = dt_test[0]
dt_prediction = decision_tree.predict(tree, dt_bird)

print(f"\n Species: {test_data[0]['species']}")
print(f"Listed Classification: {dt_bird[-1]}")
print(f"Calculated Classification: {dt_prediction}")
