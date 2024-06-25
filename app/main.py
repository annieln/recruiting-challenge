from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
from deepface import DeepFace

import numpy as np
import cv2

app = FastAPI()

profiles = []

class Profile(BaseModel):
    gender: str
    race: str
    age: float

@app.post("/create-profile", response_model=Profile)
async def create_profile(file: UploadFile = File(...), name: Optional[str] = str):
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

def analyze_face(image):
    img = convert_from_image_to_cv2(image)
    
    objs = DeepFace.analyze(
		img_path = img
	)

    return Profile(gender=objs[0]["dominant_gender"], race=objs[0]["dominant_race"], age=objs[0]["age"])

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
