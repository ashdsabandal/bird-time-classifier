# Serves as the main entry point to load data, train the three models, evaluate their performance, and compare results.

from src import utils, knn

filename = "data/dataset.csv"

bird_rows = utils.readFile(filename)
training_data, test_data = utils.splitter(bird_rows, split = 1400)

# KNN 