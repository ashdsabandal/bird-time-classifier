# Contains the implementation of the Naive Bayes classification algorithm.

from datetime import date

# naive bayes classifications 
# latitude = 4 classification (max-min)/4  by range
# longitude = 4 classification (max-min)/4 by range
# month = 12 months (by float)
# day = 31 (by float)
# duration = 4 classification 
# classification = classification

def get_ranges(bird_rows): 
  return (classify_coords(bird_rows, "latitude", bins=8), classify_coords(bird_rows, "longitude", bins=8), classify_duration(bird_rows, bins=20))

def get_splitter(values, bins=4): 
  s = sorted(values)
  return [s[len(s) * i // bins] for i in range(1, bins)]

def classify_coords(bird_rows, coordtype, bins=8):
  coords = [row[coordtype] for row in bird_rows]
  return get_splitter(coords, bins)

def classify_duration(bird_rows, bins=20): 
  durations = [row["duration"] for row in bird_rows]
  return get_splitter(durations, bins)

def get_date_fraction(month, day): 
  return date(2000, month, day).timetuple().tm_yday / 366

def get_datebin(month, day, bins=12): # for 12 months (monthly granularity)
  frac = get_date_fraction(month, day)
  return min(int(frac*bins), bins - 1)

def get_class(value, splitter): 
  return sum(value >= c for c in splitter)

def make_features(bird_rows, ranges): 
  features = []

  lat_split, long_split, dur_split = ranges

  for row in bird_rows: 
    features.append([
      row["species"],
      get_class(row["latitude"], lat_split), 
      get_class(row["longitude"], long_split), 
      get_datebin(row["month"], row["day"]),
      get_class(row["duration"], dur_split), 
      row["classification"]
    ])

  return features

def generate_bob(train_data): 
  feats =  len(train_data[0]) - 1
  bob = {} #hehe bag of birds
  # holds value of each bird for each class like {"morning": 234, "afternoon": 232, "night": 123}
  fcount = {}

  # ito yung vocabulary in bag of words setup
  unique_value = [set() for value in range(feats)]

  # for every bird in the training data
  for bird in train_data: 
    # get its label
    label = bird[-1]

    # if the label not in the bag of birds {"morning", "afternoon", "evening"}
    if label not in bob: 
      #create an entry in the dictionary with the label, containing a list of n(feat) dictionaries
      bob[label] = [{} for value in range(feats)]
      fcount[label] = 0 
    # just increment the label appeareance count
    fcount[label] += 1

    # for every feature we are monitoring (btw feature : latitude, longitude, ... )
    for i in range(feats):
      # get yung value nung feature
      value = bird[i]
      # then in bag of birds, under label, under the feature (represented by i), equate value = to the existing value (if not theree then 0) + 1
      bob[label][i][value] = bob[label][i].get(value, 0) + 1
      #add that value to unique value set for this specific feature (i)
      unique_value[i].add(value)

  # list containing counts for each unique value for each feature
  voc_count = [len(s) for s in unique_value]

  return bob, fcount, voc_count

def calc_bird_category(given_bird, bob, total, voc_count, k): 
  real = 1.0
  # formula
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

  



