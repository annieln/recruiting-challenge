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
	id: int
	filename: str
	gender: str
	race: str
	age: float
	landmarks: dict

@app.post("/profiles/")
async def create_profile(file: UploadFile = File(...)):
	"""
	Generates a new profile from an uploaded image using facial analysis.

	Parameters:
	- file (UploadFile): An image file containing a person's face.

	Returns:
	- message : A message indicating whether a profile was successfully generated or if a spoof was detected.
	"""
	if file.content_type not in ["image/jpeg", "image/png"]:
		raise HTTPException(status_code=415, detail="Unsupported file format. Please upload JPEG or PNG.")

	image = Image.open(BytesIO(await file.read()))
	img = convert_from_image_to_cv2(image)

	try:
		profile = analyze_face(img)
	except:
		return {"message" : "Spoof detected in given image."}
	else:
		profile.filename = file.filename
		profiles.append(profile)
		images.append(img)
		return {"message": f"Profile generated for {profile.filename} with Profile ID {profile.id}"}

@app.put("/profiles/{profile_id}")
async def update_profile(profile_id: int, file: UploadFile = File(...)):
	"""
	Update an existing profile with new image and information.

	Parameters:
	- profile_id (int) : The ID of the profile to be replaced.
	- file (UploadFile): An image file containing a person's face.

	Returns:
	- message : A message indicating whether a profile was successfully updated or if a spoof was detected.

	Raises:
	- HTTPException : If the profile at the given ID cannot be found.
	"""
	if profile_id < len(profiles):
		if file.content_type not in ["image/jpeg", "image/png"]:
			raise HTTPException(status_code=415, detail="Unsupported file format. Please upload JPEG or PNG.")
		
		image = Image.open(BytesIO(await file.read()))
		img = convert_from_image_to_cv2(image)

		try:
			new_profile = analyze_face(img)
		except:
			return {"message" : "Spoof detected in given image."}
		else:
			new_profile.filename = file.filename
			profiles[profile_id] = new_profile
			images[profile_id] = img
			return {"message": f"Profile {profile_id} updated successfully with new image and profile data from {new_profile.filename}."}
	else:
		raise HTTPException(status_code=404, detail=f"Profile at ID {profile_id} not found")

@app.get("/profiles/")
async def get_profiles():
	"""
	Get a list of all existing profiles.

	Returns:
	- list[Profiles] : A list of all Profile objects.
	"""
	return profiles

@app.get("/profiles/{profile_id}")
def get_profile(profile_id: int) -> Profile:
	"""
	Get a profile based on profile ID.

	Parameters:
	- profile_id (int) : The index of the profile.

	Returns:
	- Profile : The requested profile at the given ID.

	Raises:
	- HTTPException : If the profile at the given ID cannot be found.
	"""
	if profile_id < len(profiles):
		return profiles[profile_id]
	else:
		raise HTTPException(status_code=404, detail=f"Profile at ID {profile_id} not found")
	
@app.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: int):
	"""
	Delete a profile based on profile ID.

	Parameters:
	- profile_id (int) : The index of the profile.

	Returns:
	- message : A message indicating whether that the profile was deleted successfully.

	Raises:
	- HTTPException : If the profile at the given ID cannot be found.
	"""
	if profile_id < len(profiles):
		profiles.pop(profile_id)
		images.pop(profile_id)
		return {"message": f"Profile {profile_id} deleted successfully"}
	else:
		raise HTTPException(status_code=404, detail=f"Profile at ID {profile_id} not found")

@app.post("/identify/")
async def identify_profile(file: UploadFile = File(...)):
	"""
	Check if the face in an image matches any of the recorded profiles.

	Parameters:
	- file (UploadFile) : The image of a face.

	Returns:
	- Profile : The profile ID of the matching profile, if there is one.
	Otherwise, a message indicating there is no match or that a spoof
	was detected in the uploaded image.
	"""
	if file.content_type not in ["image/jpeg", "image/png"]:
		raise HTTPException(status_code=415, detail="Unsupported file format. Please upload JPEG or PNG.")
	
	image = Image.open(BytesIO(await file.read()))
	img = convert_from_image_to_cv2(image)

	for id in range(len(profiles)):
		src = images[id]
		try:
			result = DeepFace.verify(src, img, anti_spoofing=True)
		except:
			return {"result" : "Spoof detected in given image."}
		if result["verified"]:
			return {"result" : id}
	return {"result" : "No matching profile found"}

@app.get("/profiles/{profile_id}/landmarks")
async def get_landmarks(profile_id: int):
	"""
	Get the facial landmarks of the profile given by the profile ID.

	Parameters:
	- profile_id (int) : The index of the profile.

	Returns:
	- landmarks (dict) : A dictionary where keys are the facial features and 
	values are the associated points within the image.

	Raises:
	- HTTPException : If the profile at the given ID cannot be found.

	"""
	if profile_id < len(profiles):
		profile = profiles[profile_id]
		return profile.landmarks
	else:
		raise HTTPException(status_code=404, detail=f"Profile at ID {profile_id} not found")
	
@app.get("/profiles/{profile_id}/age")
async def get_age(profile_id: int):
	"""
	Get the age of the profile given by the profile ID.

	Parameters:
	- profile_id (int) : The index of the profile.

	Returns:
	- age (int) : The estimated age of the face in the profile.

	Raises:
	- HTTPException : If the profile at the given ID cannot be found.

	"""
	if profile_id < len(profiles):
		profile = profiles[profile_id]
		return profile.age
	else:
		raise HTTPException(status_code=404, detail=f"Profile at ID {profile_id} not found")
	
@app.get("/profiles/{profile_id}/gender")
async def get_gender(profile_id: int):
	"""
	Get the gender of the profile given by the profile ID.

	Parameters:
	- profile_id (int) : The index of the profile.

	Returns:
	- gender (int) : The estimated gender of the face in the profile.

	Raises:
	- HTTPException : If the profile at the given ID cannot be found.
	"""
	if profile_id < len(profiles):
		profile = profiles[profile_id]
		return profile.age
	else:
		raise HTTPException(status_code=404, detail=f"Profile at ID {profile_id} not found")
	
@app.get("/profiles/{profile_id}/race")
async def get_race(profile_id: int):
	"""
	Get the race of the profile given by the profile ID.

	Parameters:
	- profile_id (int) : The index of the profile.

	Returns:
	- race (int) : The estimated race of the face in the profile.

	Raises:
	- HTTPException : If the profile at the given ID cannot be found.
	"""
	if profile_id < len(profiles):
		profile = profiles[profile_id]
		return profile.race
	else:
		raise HTTPException(status_code=404, detail=f"Profile at ID {profile_id} not found")

#### Helper functions ####

def analyze_face(image):
	"""
	Analyze the face in a given image and returns a Profile.

	Detects whether the image is a spoof and performs facial analysis.
	"""

	objs = DeepFace.analyze(
		img_path = image,
		anti_spoofing = True
	)

	faces, features = detect_face(image)

	return Profile(id=len(profiles), filename="", gender=objs[0]["dominant_gender"], race=objs[0]["dominant_race"], age=objs[0]["age"], landmarks=features)

def detect_face(image):
	"""
	Performs facial analysis using Haar Cascade algorithm for
	face detection and LBF for facial landmark extraction.
	"""
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
	"""
	Converts an image to an accepted format for OpenCV.
	"""
	return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)