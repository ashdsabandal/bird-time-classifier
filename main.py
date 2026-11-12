# Serves as the main entry point to load data, train the three models, evaluate their performance, and compare results.

from src import utils, knn, naive_bayes, decision_tree

filename = "data/dataset.csv"

bird_rows = utils.readFile(filename)
training_data, test_data = utils.splitter(bird_rows, split = 4800)

# KNN 
  # FOR JOEY AND ASH: BAKA NEED NYO RIN GAWIN TONG STUFF NA TO KASI 
  # FOR FORMATTING SYA , ESPECIALLY SI SCALE FEATURES 

print("\n KNN")
knn_train = knn.month_to_float(training_data)
knn_test = knn.month_to_float(test_data)

knn_train, knn_test = utils.scale_features(knn_train, knn_test, skip=[3])

# k = int(input("Enter K: "))
# p = float(input("Enter p: "))
k = 5
p = 2

bird = knn_test[0]
prediction = knn.predict(k,p,knn_train, bird)

print(f"\n Species: {test_data[0]['species']}")
print(f"Listed Classification: {bird[-1]}")
print(f"Calculated Classification: {prediction}")

# NAIVE BAYES
print("\n NAIVE BAYES")

ranges = naive_bayes.get_ranges(training_data)

nb_train = naive_bayes.make_features(training_data, ranges)
nb_test = naive_bayes.make_features(test_data, ranges)

bob, fcount, voc_count = naive_bayes.generate_bob(nb_train)

for label in fcount:
  print(label)
  print(f"Number of Birds: {fcount[label]}")
  print("\n")

print(f"Training Vocabulary (per feature): {voc_count}")

for label in fcount:
  print(f"P({label}): {fcount[label]/len(nb_train)}")

k_val = int(input("Enter k: "))
while k_val <= 0:
  print("please put a positive num")
  k_val = int(input("Enter k: "))

bird = nb_test[0]
prediction, probs = naive_bayes.predict(k_val, bob, fcount, voc_count, bird)

print("\nClass Probabilities:")
for label in probs:
  print(f"{label}: {probs[label]:.4f}")

print(f"\nSpecies: {test_data[0]['species']}")
print(f"Actual classification: {bird[-1]}")
print(f"Predicted classification: {prediction}")

# Decision Tree (CART)
print("\n DECISION TREE (CART)")
dt_train = decision_tree.format_rows(training_data)
dt_test = decision_tree.format_rows(test_data)

tree = decision_tree.build_tree(dt_train, max_depth=5, min_size=10)
dt_bird = dt_test[0]
dt_prediction = decision_tree.predict(tree, dt_bird)

print(f"\n Species: {test_data[0]['species']}")
print(f"Listed Classification: {dt_bird[-1]}")
print(f"Calculated Classification: {dt_prediction}")
