#!python2
from sys import argv
import numpy as np
import os
from pyAudioAnalysis import audioTrainTest as aT
script, filedir = argv

filename = os.listdir(filedir)
for i in filename:
  Result, P, classNames = aT.fileClassification(filedir + "/" + i, "svmModel", "svm")
  print(i)
  winner = np.argmax(P) 

  win = i + "	" + str(P[winner]) + "	" + classNames[winner]

  f = open("result.txt", "a")
  f.write(win + "\n")
  f.close()
