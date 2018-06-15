import cv2
import numpy

FACE_RATE = 0.15

def get_faces(img):
	model_path = "./haarcascade_frontalface_default.xml"
	img_width = len(img[0])
	img_height = len(img)

	face_limit = 0
	if img_width > img_height:
		face_limit = int(img_height * FACE_RATE)
	else:
		face_limit = int(img_width * FACE_RATE)

	face_cascade = cv2.CascadeClassifier(model_path)
	try:
		faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(face_limit,face_limit), flags=cv2.CASCADE_SCALE_IMAGE)
		for face in faces:
				face[2] += face[0]
				face[3] += face[1]
	except Exception as msg:
		faces = []
	
	"""
	if len(faces) > 0 : 
		for (x,y,w,h) in faces:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	"""
	return faces

