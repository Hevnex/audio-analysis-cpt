#!python2
from sys import argv
import numpy as np
import os
from pyAudioAnalysis import audioTrainTest as aT
script, filedir = argv
isSignificant = 0.3 #try different values.

filename = os.listdir(filedir)
for i in filename:
  # P: list of probabilities
  Result, P, classNames = aT.fileClassification(filedir + "/" + i, "svmModel", "svm")
  print(i)
  winner = np.argmax(P) #pick the result with the highest probability value.

# is the highest value found above the isSignificant threshhold? 
  if P[winner] > isSignificant :
    win = i + "	" + str(P[winner])[:5] + "	" + classNames[winner]
  else :
    win = i + "	" + "-"

  f = open("result.txt", "a")
  f.write("\n" + win)
  f.close()
