# Contains the implementation of the Naive Bayes classification algorithm.

# naive bayes classifications 
# latitude = 4 classification (max-min)/4  by range
# longitude = 4 classification (max-min)/4 by range
# month = 12 months (by float)
# day = 31 (by float)
# duration = 4 classification 
# classification = classification

def get_ranges(bird_rows): 
  return( classify_coords(bird_rows, "latitude"), classify_coords(bird_rows, "longitude"), classify_duration(bird_rows))

def classify_coords(bird_rows, coordtype):
  coords = [row[coordtype] for row in bird_rows]
  maximum, minimum = max(coords), min(coords)

  return maximum, minimum

def classify_duration(bird_rows): 
  durations = [row["duration"] for row in bird_rows]
  maximum, minimum = max(durations), min(durations)

  return maximum, minimum

def get_class(coord, max, min): 

  n = (max-min)/4

  if coord < min+n: return 0
  if min+n <= coord < min+2*n: return 1
  if min+2*n <= coord < min+3*n: return 2
  
  return 3

def get_day(day): 
  if day < 8: return 0
  if 8 <= day < 15: return 1 
  if 15 <= day < 22: return 2

  return 3
  
def make_features(bird_rows, ranges): 
  features = []

  (max_lat, min_lat), (max_long, min_long), (max_dur, min_dur) = ranges

  for row in bird_rows: 
    features.append([
      get_class(row["latitude"], max_lat, min_lat), 
      get_class(row["longitude"], max_long, min_long), 
      float(row["month"]), 
      get_day(row["day"]),
      get_class(row["duration"], max_dur, min_dur), 
      row["classification"]
    ])

  return features

def generate_bob(train_data): 
  feats =  len(train_data[0]) - 1
  bob = {} #hehe bag of birds
  fcount = {}

  unique_value = [set() for value in range(feats)]

  for bird in train_data: 
    label = bird[-1]

    if label not in bob: 
      bob[label] = [{} for value in range(feats)]
      fcount[label] = 0 

    fcount[label] += 1

    for i in range(feats):
      value = bird[i]
      bob[label][i][value] = bob[label][i].get(value, 0) + 1
      unique_value[i].add(value)

  voc_count = [len(s) for s in unique_value]

  return bob, fcount, voc_count

def calc_bird_category(given_bird, bob, total, voc_count, k): 
  real = 1.0

  for i in range(len(voc_count)): 
    count = bob[i].get(given_bird[i], 0)
    real *= (count + k) / (total + k * voc_count[i])

  return real

def predict(k_val, bob, fcount, voc_count, given_bird): 
  total_birds = sum(fcount.values())
  p_class = {}
  p_bird_class = {}
  p_bird = 0.0

  for label in fcount: 
    p_class[label] = fcount[label] / total_birds
    p_bird_class[label] = calc_bird_category(given_bird, bob[label], fcount[label], voc_count, k_val)
    p_bird += p_bird_class[label] * p_class[label]

  probs = {}
  classification = ""
  calculation = 0.0

  for label in fcount: 
    p_category = p_bird_class[label] * p_class[label] / p_bird
    probs[label] = p_category

    if p_category > calculation: 
      classification = label
      calculation = p_category

  return classification, probs

  



