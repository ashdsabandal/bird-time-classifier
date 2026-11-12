# Contains the implementation of the k-Nearest Neighbors (k-NN) classification algorithm.

from datetime import date
#makes it so values si month and day, tapos yung features and classification lang remove the bird names and other irrelevant data

def get_date_fraction(month, day): 
  return date(2000, month, day).timetuple().tm_yday / 366

def month_to_float(bird_rows): 
  features = []

  for r in bird_rows: 
    features.append([
      r["species"],
      r["latitude"], 
      r["longitude"], 
      get_date_fraction(r["month"], r["day"]),
      r["duration"], 
      r["classification"]
    ])

  return features

def compute_minowski(given, bird, p, weights=None): 
  if weights is None:
    # Feature weights: species (lowered penalty), latitude, longitude, date_fraction, duration (elevated)
    weights = (0.1, 1.0, 1.0, 1.0, 2.5)
  total = 0.0 

  for i, (x, y) in enumerate(zip(given[0:-1], bird[0:-1])): 
    w = weights[i] if i < len(weights) else 1.0
    if isinstance(x, str):
      diff = 0.0 if x == y else 1.0
    elif i == 3: 
      d = abs(x - y)
      diff = min(d, 1.0 - d)
    else:
      diff = abs(x - y)

    scaled_diff = diff * w
    if p == 2:
      total += scaled_diff * scaled_diff
    else:
      total += pow(scaled_diff, p)

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

  # Distance-weighted voting reduces bias from majority class ties
  for distance, label in neighbors[0:k]: 
    weight = 1.0 / (distance + 0.001)
    sorted_time[label] = sorted_time.get(label, 0.0) + weight

  return max(sorted_time, key=sorted_time.get)

