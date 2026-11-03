# Loads data from CSV, handles missing values, feature scaling/encoding, and partitions data into train/test splits.

import math
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

  return rows

def split_data(bird_rows, split): 
  return bird_rows[:split], bird_rows[split:]

# FOR DATA SCALING TO MAKE SURE THEY'RE BALANCED 
#scaling data 
def scale_row(r, min, max): 
  scaled_row = []

  for i in range(len(min)): 
    if min[i] != max[i]: 
      scaled_row.append((r[i] - min[i])/ (max[i] - min[i]))
    else: 
      scaled_row.append(0.0)

  scaled_row.append(r[-1])

  return scaled_row

def scale_features(train, test): 
  n = len(train[0]) - 1 #remove si "afternoon", "morning", "evening"
  min = [min(row[feature] for row in train) for feature in range(n)]
  max = [max(row[feature] for row in train) for feature in range(n)]

  scale_train = [scale_row(r) for r in train]
  scale_test = [scale_row(r) for r in test]

  return scale_train, scale_test