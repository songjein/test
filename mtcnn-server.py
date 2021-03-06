from flask import Flask
from flask import jsonify, send_from_directory, request
from flask_cors import CORS

import uuid

from mtcnn.mtcnn import MTCNN
import fd_cv2 

import cv2
import time

from os import listdir
from os.path import isfile, join

import numpy as np

from skimage import io


app = Flask(__name__)
cors = CORS(app)

detector = MTCNN()

@app.route("/facethumb", methods=["POST"])
def facethumb():
		root = "http://" + request.remote_addr
		port = request.form["port"]
		api = request.form["api"]
		filename = request.form["filename"]

		# download image
		print( "api", root +  ":" + port +  api + filename, request.remote_addr)
		image = io.imread(root + ":" + port + api + filename) # rgb
		cimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr

		# MTCNN  
		stime = time.time();
		faces = (detector.detect_faces(image))
		mtcnn_time = time.time() - stime
		meta = {"faces": faces, "time": mtcnn_time}

		for f in faces:
				cv2.rectangle(
					cimage # bgr
					,(f["box"][0], f["box"][1]) 
					,(f["box"][0] + f["box"][2], f["box"][1] + f["box"][3]),
					(0,0,255), 2);

		cv2.imwrite("./tmp/%s" %(filename), cimage);
		
		# openCV
		stime = time.time()
		cimage_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		faces_cv = fd_cv2.get_faces(image)
		opencv_time = time.time() - stime

		for f in faces_cv:
				cv2.rectangle(
					cimage_cv
					,(f[0], f[1]) 
					,(f[2], f[3]),
					(0,0,255), 2);
		meta["time_cv"] = opencv_time

		cv2.imwrite("./tmp/cv_%s" %(filename), cimage_cv);

		return jsonify({"api": "http://192.168.182.195:22222/photo", "filename": filename, "meta": meta})

@app.route("/facethumb_via_url", methods=["POST"])
def facethumb_via_url():
		url = request.form["url"]
		filename = str(uuid.uuid4()) + ".jpg"

		# download image
		print("facethumb via url", url)
		image = io.imread(url)
		cimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# MTCNN  
		stime = time.time();
		faces = (detector.detect_faces(image))
		mtcnn_time = time.time() - stime
		meta = {"faces": faces, "time": mtcnn_time}

		for f in faces:
				cv2.rectangle(
					cimage
					,(f["box"][0], f["box"][1]) 
					,(f["box"][0] + f["box"][2], f["box"][1] + f["box"][3]),
					(0,0,255), 2);

		cv2.imwrite("./tmp/%s" %(filename), cimage);
		
		# openCV
		stime = time.time()
		cimage_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		faces_cv = fd_cv2.get_faces(image)
		opencv_time = time.time() - stime

		for f in faces_cv:
				cv2.rectangle(
					cimage_cv
					,(f[0], f[1]) 
					,(f[2], f[3]),
					(0,0,255), 2);
		meta["time_cv"] = opencv_time

		cv2.imwrite("./tmp/cv_%s" %(filename), cimage_cv);

		return jsonify({"api": "http://192.168.182.195:22222/photo", "filename": filename, "meta": meta})



@app.route("/photo/<path:path>")
def send_photo(path):
		return send_from_directory('tmp', path)

'''
#img_path = "/home/image/sample/설현/"
img_path = "./"
#files = [img_path + f for f in listdir(img_path) if isfile(join(img_path, f))]
files = ["test18.jpg", "test17.jpg", "test16.jpg", "test15.jpg", "test14.jpg", "test13.jpg", "test12.jpg", "test11.jpg", "test1.jpg", "test2.jpg", "test5.jpg","test6.jpg","test7.jpg","test8.jpg","test9.jpg","test10.jpg"]

for i in range(len(files)):
  f = files[i]
  stime = time.time()
  image = cv2.imread(f)
  faces = (detector.detect_faces(image))
  
  for f in faces:
    cv2.rectangle(
      image
      ,(f["box"][0], f["box"][1]) 
      ,(f["box"][0] + f["box"][2], f["box"][1] + f["box"][3]),
      (0,0,255), 2);

  cv2.imwrite("./tmp/result_%d.jpg" %(i), image);
'''
