# Loads data from CSV, handles missing values, feature scaling/encoding, and partitions data into train/test splits.

import random
import csv
from datetime import datetime

# formating: 
  # bird_rows = [{'species': string, 'latitutde': float, 'longitude': float, 'month': int, 'day': int, 'duration': float, 'classification': string}, 
  #              {next row...}, ...]


def time_sorter(hour): 
  if 4 <= hour < 12: 
    return "Morning"
  if 12 <= hour < 17: 
    return "Afternoon"

  return "Night"

def readFile(filename): 
  rows = []

  with open(filename, newline="", encoding = "utf-8") as f: 
    textreader = csv.reader(f)
    next(textreader) #basta to skip si header 

    # name = 0 
    # latitude = 4 , longtitude = 5
    # date = 6, time = 7, duration = 8

    for r in textreader: 
      if not r[7] or not r[8]: 
        continue
      date = datetime.strptime(r[6], "%Y-%m-%d")
      hour = int(r[7].split(":")[0])

      rows.append({
        "species": r[0], 
        "latitude": float(r[4]), 
        "longitude": float(r[5]), 
        "month": date.month, 
        "day": date.day, 
        "duration": float(r[8]), 
        "classification": time_sorter(hour)
      })

  #shuffles the order para hindi arragned by date yung dataset natin
  random.seed(67)
  random.shuffle(rows)

  return rows

def splitter(rows, split):
  groups = {}

  for r in rows:
    key = (r["latitude"], r["longitude"], r["month"], r["day"], r["duration"])
    if key in groups:
      groups[key].append(r)
    else:
      groups[key] = [r]

  train, test = [], []
  for key in groups:
    if len(train) < split:
      train.extend(groups[key])
    else:
      test.extend(groups[key])

  return train, test

# FOR DATA SCALING TO MAKE SURE THEY'RE BALANCED 
# https://towardsdatascience.com/scaling-numerical-data-explained-a-visual-guide-with-code-examples-for-beginners-11676cdb45cb/
#scaling data 
def scale_row(r, mins, maxs): 
  scaled_row = []

  for i in range(len(mins)): 
    if mins[i] != maxs[i]: 
      scaled_row.append((r[i] - mins[i])/ (maxs[i] - mins[i]))
    else: 
      scaled_row.append(0.0)

  scaled_row.append(r[-1])

  return scaled_row

def scale_features(train, test): 
  n = len(train[0]) - 1 #remove si "afternoon", "morning", "evening"
  mins = [min(row[feature] for row in train) for feature in range(n)]
  maxs = [max(row[feature] for row in train) for feature in range(n)]

  print(mins)
  print(maxs)
  scale_train = [scale_row(r, mins, maxs) for r in train]
  scale_test = [scale_row(r, mins, maxs) for r in test]

  return scale_train, scale_test