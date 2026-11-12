# Contains the implementation of the k-Nearest Neighbors (k-NN) classification algorithm.


#makes it so values si month and day, tapos yung features and classification lang remove the bird names and other irrelevant data
def month_to_float(bird_rows): 
  features = []

  for r in bird_rows: 
    features.append([
      r["species"],
      r["latitude"], 
      r["longitude"], 
      float(r["month"]), 
      float(r["day"]), 
      r["duration"], 
      r["classification"]
    ])

  return features

def compute_minowski(given, bird, p): 
  total = 0 

  for x, y in zip(given[0:-1], bird[0:-1]): 
    if isinstance(x, str):
      diff = 0.0 if x == y else 1.0
    else:
      diff = abs(x-y)
    total += pow(diff, p)

  return pow(total, 1/p)

def predict(k, p, train_data, given_bird): 
  neighbors = []

  for bird in train_data: 
    distance = compute_minowski(given_bird, bird, p)
    neighbors.append((distance, bird[-1]))

  neighbors.sort(key=lambda x: x[0])

  sorted_time = {}

  print(f"\n Closest {k} neighbors with p = {p}")
  print("NO.  DISTANCE  LABEL")
  rank = 1
  for distance, label in neighbors[0:k]: 
    print(f"{rank}  {distance}  {label}")
    rank += 1

  for distance, label in neighbors[0:k]: 
    if label in sorted_time: 
      sorted_time[label] += 1
    else: 
      sorted_time[label] = 1

  return max(sorted_time, key=sorted_time.get)

