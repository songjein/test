from mtcnn.mtcnn import MTCNN
import cv2
import time

import fd_cv2 

from os import listdir
from os.path import isfile, join

import numpy as np

detector = MTCNN()

img_path = "/home/image/sample/설현/"
#img_path = "./imgs/설현/"
#img_path = "./"
files = [img_path + f for f in listdir(img_path) if isfile(join(img_path, f))]
#files = ["test1.jpg"]

"""
############# GETSIZE ################
sizes = []
for i in range(len(files)):
	image = cv2.imread(files[i])
	print (image.shape[:-1])
	
############# GETSIZE ################
"""

#mode = "CNN"
mode = "CV"
diffs = []
faceCnts = []
THRES = 9999 

print (files[:5])
for i in range(len(files)):
		try:
				#f = "tmp/" + files[i]
				#f = "./" + files[i]

				stime = time.time()
				image = cv2.imread(files[i])
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # rgb
				if (image.shape[1] > THRES):
						ratio = THRES/image.shape[1]
						image = cv2.resize(image, (THRES, int(image.shape[0] * ratio)))

				if (mode == "CV"):
						faces = fd_cv2.get_faces(image)
						diff = (time.time() - stime) * 1000
						diffs.append(diff)
						faceCnts.append(len(faces))
						#print (faces, diff)
						'''
						for f in faces:
								cv2.rectangle(
									image
									,(f[0], f[1]) 
									,(f[2], f[3]),
									(0,0,255), 2);
									'''

				if (mode == "CNN"):
						faces = (detector.detect_faces(image))
						diff = (time.time() - stime) * 1000
						diffs.append(diff)
						faceCnts.append(len(faces))
						#print (faces, diff)
						'''
						for f in faces:
								cv2.rectangle(
									image
									,(f["box"][0], f["box"][1]) 
									,(f["box"][0] + f["box"][2], f["box"][1] + f["box"][3]),
									(0,0,255), 2);
						'''
				if (i % 100 == 0):
					print (i)


				#cv2.imwrite("./tmp/result_%d.jpg" %(i), image);
				#cv2.imwrite("./result_%d.jpg" %(i), image);
		except Exception as e:
			print(files[i], e)
np.save("cv_cpu_result_times_%s.npy" %(THRES), arr=diffs)
np.save("cv_cpu_result_fcnts_%s.npy" %(THRES), arr=faceCnts)
