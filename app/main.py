from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
from deepface import DeepFace
from collections import OrderedDict

import numpy as np
import cv2

app = FastAPI()

LBFmodel = "lbfmodel.yaml"
profiles = []
images = []

facial_landmarks_idx = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
	])

class Profile(BaseModel):
	index: int
	gender: str
	race: str
	age: float
	features: dict


@app.post("/create-profile", response_model=Profile)
async def create_profile(file: UploadFile = File(...)):
	"""
	Create a profile.

	Parameters:
		file (UploadFile) : The image to create profile from.

	Returns:
		Profile : The created profile.
	"""

	image = Image.open(BytesIO(await file.read()))
	profile = analyze_face(image)
	profiles.append(profile)

	return profile

@app.get("/get-profile/{profile_id}", response_model=Profile)
def get_profile(profile_id: int) -> Profile:
	"""
	Get a profile.

	Parameters:
		profile_id (int) : The index of the profile.

	Returns:
		Profile : The requested profile.
	"""
	if profile_id < len(profiles):
		return profiles[profile_id]
	else:
		raise HTTPException(status_code=404, detail=f"Profile {profile_id} not found")

@app.post("/match-profile", response_model=Profile)
async def match_profile(file: UploadFile = File(...)):
	"""
	Check if the face in an image matches any of the recorded profiles.

	Parameters:
		file (UploadFile) : The image of a face.

	Returns:
		Profile : The matching profile, if there is one.
	"""
	image = Image.open(BytesIO(await file.read()))
	img = convert_from_image_to_cv2(image)

	for profile in profiles:
		src = images[profile.index]
		result = DeepFace.verify(src, img)
		if result["verified"]:
			return profile
	raise HTTPException(status_code=404, detail="No matching profile found")

	#### Helper functions ####

def analyze_face(image):
	img = convert_from_image_to_cv2(image)

	faces, features = detect_face(img)

	objs = DeepFace.analyze(
		img_path = img
	)

	return Profile(index=len(profiles), gender=objs[0]["dominant_gender"], race=objs[0]["dominant_race"], age=objs[0]["age"], features=features)

def detect_face(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	face_classifier = cv2.CascadeClassifier(
		cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
	)

	faces = face_classifier.detectMultiScale(
		gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
	)

	landmark_detector = cv2.face.createFacemarkLBF()
	landmark_detector.loadModel(LBFmodel)
	_, landmarks = landmark_detector.fit(image, faces)

	facial_features = {}

	for (i, name) in enumerate(facial_landmarks_idx.keys()):
		(j, k) = facial_landmarks_idx[name]
		pts = landmarks[0][0][j:k]
		facial_features[name] = pts.tolist()
	
	return faces, facial_features

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
	return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# def detect_landmarks(image):
# 	img = convert_from_image_to_cv2(image)
# 	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 	face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# 	faces = face_classifier.detectMultiScale(
# 		gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
# 	)

# 	for (x, y, w, h) in faces:
# 		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# 	landmark_detector = cv2.face.createFacemarkLBF()
# 	landmark_detector.loadModel(LBFmodel)
# 	_, landmarks = landmark_detector.fit(img, faces)
# 	print(len(landmarks[0][0]))

# 	colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
# 			(168, 100, 168), (158, 163, 32),
# 			(163, 38, 32), (180, 42, 220)]

# 	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
# 		(j, k) = FACIAL_LANDMARKS_IDXS[name]
# 		pts = landmarks[0][0][j:k]
# 		print(pts)
# 		if name == "jaw":
# 			for l in range(1, len(pts)):
# 				ptA = tuple([int(pts[l - 1][0]), int(pts[l - 1][1])])
# 				print(ptA)
# 				ptB = tuple([int(pts[l][0]), int(pts[l][1])])
# 				print(ptB)
# 				cv2.line(img, ptA, ptB, colors[i], 2)
# 		else:
# 			hull = cv2.convexHull(pts)
# 			cv2.drawContours(img, np.int32([hull]), -1, colors[i], -1)

# 	for landmark in landmarks:
# 		for x,y in landmark[0]:
# 			# display landmarks on "image_cropped"
# 			# with white colour in BGR and thickness 1
# 			cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 4)
# 	cv2.imwrite("written.png", img)