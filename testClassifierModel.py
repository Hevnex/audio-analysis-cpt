#!/usr/local/bin/python2
from sys import argv
import numpy as np
from pyAudioAnalysis import audioTrainTest as aT
script, filename = argv
isSignificant = 0.6 #try different values.

# P: list of probabilities
Result, P, classNames = aT.fileClassification(filename, "svmModel", "svm")
winner = np.argmax(P) #pick the result with the highest probability value.

if P[winner] > isSignificant :
  print("File: " +filename + " is in category: " + classNames[winner] + ", with probability: " + str(P[winner]))
else :
  print("Can't classify sound: " + str(P))