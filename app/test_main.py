import pytest
import numpy as np
from fastapi.testclient import TestClient
from main import app, profiles, Profile

client = TestClient(app)

# Load an actual test image
with open("test_image.jpg", "rb") as image_file:
    TEST_IMAGE = image_file.read()

with open("test_image_2.jpg", "rb") as image_file:
    TEST_NO_MATCH = image_file.read()

with open("test_fake.jpeg", "rb") as image_file:
    TEST_FAKE = image_file.read()

def test_create_profile():
    response = client.post(
        "/profiles/",
        files={"file": ("test_image.jpg", TEST_IMAGE, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["message"].startswith("Profile generated for test_image.jpg")

def test_create_profile_spoof():
    response = client.post(
        "/profiles/",
        files={"file": ("test_fake.jpg", TEST_FAKE, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "Spoof detected in given image."

def test_get_profiles():
    response = client.get("/profiles/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_profile():
    response = client.get("/profiles/0")
    assert response.status_code == 200
    assert response.json()["id"] == 0

def test_update_profile():
    response = client.put(
        "/profiles/0",
        files={"file": ("test_update.jpg", TEST_IMAGE, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["message"].startswith("Profile 0 updated successfully")

def test_delete_profile():
    response = client.delete("/profiles/0")
    assert response.status_code == 200
    assert response.json()["message"] == "Profile 0 deleted successfully"

def test_identify_profile():
    response = client.post(
        "/profiles/",
        files={"file": ("test_image.jpg", TEST_IMAGE, "image/jpeg")}
    )
        
    response = client.post(
        "/identify/",
        files={"file": ("test_identify.jpg", TEST_IMAGE, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["result"] == 0

def test_identify_no_match():
    response = client.post(
        "/profiles/",
        files={"file": ("test_image.jpg", TEST_IMAGE, "image/jpeg")}
    )
    
    response = client.post(
        "/identify/",
        files={"file": ("test_identify_no_match.jpg", TEST_NO_MATCH, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["result"] == "No matching profile found"

def test_identify_spoof():
    response = client.post(
        "/profiles/",
        files={"file": ("test_image.jpg", TEST_IMAGE, "image/jpeg")}
    )
        
    response = client.post(
        "/identify/",
        files={"file": ("test_identify_spoof.jpeg", TEST_FAKE, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["result"] == "Spoof detected in given image."

def test_get_landmarks():
    response = client.get("/profiles/0/landmarks")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

def test_get_age():
    response = client.get("/profiles/0/age")
    assert response.status_code == 200
    assert isinstance(response.json(), (float, int))

def test_get_race():
    response = client.get("/profiles/0/race")
    assert response.status_code == 200
    assert isinstance(response.json(), str)