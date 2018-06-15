from mtcnn.mtcnn import MTCNN
import cv2
import time

import fd_cv2 

from os import listdir
from os.path import isfile, join

import numpy as np

img_path = "/home/image/sample/설현/"
#img_path = "./"
files = np.array([img_path + f for f in listdir(img_path) if isfile(join(img_path, f))])
print(files[:5])

l = [200,
 247,
 340,
 395,
 402,
 422,
 437,
 454,
 557,
 582,
 898,
 910,
 919,
 998,
 1104,
 1127,
 1218,
 1457,
 1607,
 1682,
 1717,
 1720,
 1721,
 1849,
 1948,
 2040,
 2250,
 2275,
 2487,
 2537,
 2601,
 2677,
 2881,
 2898,
 2957,
 3077,
 3174,
 3176,
 3241,
 3259,
 3274,
 3316,
 3377,
 3414,
 3511,
 3573,
 3603,
 3680,
 3681,
 3726,
 3772,
 3848,
 3871,
 3953,
 3973,
 4022,
 4319,
 4325,
 4362,
 4401,
 4408,
 4458,
 4473,
 4477,
 4510,
 4608,
 4737,
 4745,
 5260,
 5360,
 5387,
 5415,
 5445,
 5467,
 5611,
 5673,
 5729,
 5747,
 5758,
 5833,
 6007,
 6295,
 6401,
 6541,
 6580,
 6587,
 6676
]

#print (files[l])

np.save("oneperson.npy" , arr=files[l])
