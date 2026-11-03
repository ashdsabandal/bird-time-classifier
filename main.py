# Serves as the main entry point to load data, train the three models, evaluate their performance, and compare results.
import math
import csv
from datetime import datetime

def time_sorter(hour): 
  if 4 <= hour <= 12: 
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
      date = datetime.strptime(r[6], "%Y-%m-%d")
      hour = int(r[7].split(":")[0])

      rows.append({
        "species": r[0], 
        "latitutde": float(r[4]), 
        "longitude": float(r[5]), 
        "month": date.month, 
        "day": date.day, 
        "duration": float(r[8]), 
        "classification": time_sorter(hour)
      })

  return rows


filename = "data/dataset.csv"
bird_rows = readFile(filename)

# formating: 
  # bird_rows = [{'species': string, 'latitutde': float, 'longitude': float, 'month': int, 'day': int, 'duration': float, 'classification': string}, 
  #              {next row...}, ...]

print(bird_rows[0])