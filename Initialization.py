#Download the vgg16.h5 and vgg16_bn.h5 files from http://files.fast.ai/models/ and paste in .keras/models folder

%matplotlib inline
import sys
print (sys.version) 
import os
print(os.getcwd())
# Create folder in cwd as below
path = "data/dogbreeds/"

from __future__ import division,print_function
import os, json
from glob import glob
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt

# Import Theano after changing tf to th in keras.json file
# Theano
import theano
print('Theano: %s' % theano.__version__)
import keras
print('Keras: %s' % keras.__version__)

#Import the utils file from the working directory.
import utils; reload(utils)
from utils import plots

# Import our class, and instantiate
import vgg16; reload(vgg16)
from vgg16 import Vgg16
vgg = Vgg16()

#Download a set of test pictures and drop in a folder
#Possibly carry out preprocessing on images before parsing them to the vgg model
batches = vgg.get_batches(path+'test5', batch_size=1032)
imgs,labels = next(batches)
prediction1 = vgg.predict(imgs, True)

#Store the output of prediction in a .csv file 
import sys
orig_stdout = sys.stdout
f = open(path+'Results\Prediction4.csv', 'w',)
sys.stdout = f

import string
for item in prediction1:
  temp = (','.join(str(s) for s in item) + '\n') 
  print(string.replace(temp, '\n', ''))

sys.stdout = orig_stdout
f.close()
