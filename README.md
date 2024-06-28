# IdentifAI Recruiting Challenge

## Overview
This FastAPI application allows users to upload an image and generate a facial profile that locates key facial landmarks and estimates demographics associated with the face. It supports creating, updating, retrieving, and deleting facial profiles and their attributes, as well as identifying faces that match existing generated facial profiles.

## Setup Instructions

### Prerequisites
- Python 3.8+
- FastAPI
- Uvicorn
- Pydantic
- Pillow
- OpenCV
- DeepFace
- NumPy

See `requirements.txt`

### Quick Start Guide
1. Clone the repository.
```bash
git https://github.com/annieln/recruiting-challenge.git
cd recruiting-challenge
```

2. Install dependencies.
```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:
```bash
cd app
uvicorn main:app --reload
```

### Basic Usage
1. Open your browser and navigate to `http://127.0.0.1:8000/docs` to access the Swagger UI interactive API documentation.
2. Use the `/profiles` endpoints to create, update, retrieve, and delete facial profiles.
3. Use the `identify` endpoint to identify new profiles against existing profiles.

### API Endpoints Overview
The following endpoints are available in the application:
- __POST /profiles/__: Create a new profile.
- __GET /profiles/__: Retrieve all profiles.
- __PUT /profiles/`{profile_id}`__: Update an existing profile.
- __GET /profiles/`{profile_id}`__: Retrieve a specific profile.
- __DELETE /profiles/`{profile_id}`__: Delete a specific profile.
- __POST /identify/__: Identify a profile based on an uploaded image.
- __GET /profiles/`{profile_id}`/landmarks__: Get the facial landmarks of a specific profile.
- __GET /profiles/`{profile_id}`/age__: Get the estimated age of a specific profile.
- __GET /profiles/`{profile_id}`/gender__: Get the estimated gender of a specific profile.
- __GET /profiles/`{profile_id}`/race__: Get the estimated race of a specific profile.

For detailed API documentation, see the interactive API documentation at `http://127.0.0.1:8000/docs`.

## Design Decisions

### Technologies and External Libraries
__OpenCV__ is utilized for its robust capabilities in image processing and facial detection. This project uses OpenCV's built-in Haar-Cascades for detecting faces within images and LBF (Local Binary Patterns Cascades for Face Recognition) model for precise of detection, allowing for detailed facial feature extraction. Furthermore, OpenCV integrates well with other libraries and frameworks including DeepFace, which uses OpenCV as its default face detector.

__DeepFace__ is used for its facial recognition and verification services. It uses pre-trained models to perform tasks such as age estimation, gender detection, and race estimation, which helps generate a generic persona for the detected face in an uploaded image. This FastAPI application also employs DeepFace for its face anti-spoofing analysis, which is vital in detecting real images from fake ones.

### Data Storage and Management
The project utilizes in-memory data structures for simplicity, as it allows for quick access and manipulation of data, which is suitable for the initial development and deployment process of this small-scale application. However, data is lost when the server is restarted, so a future consideration for persistent and scalable data storage would involve using a database, e.g. SQLite, PostgreSQL.

- The `profiles` list as an in-memory repository for storing `Profile` objects generated during the application's lifecycle. Each `Profile` includes essential descriptors such as id, filename, gender, race, age, and landmarks.
- The `image` list stores process processed images converted to OpenCV format. This facilitates rapid access and manipulation of image data for facial analysis and comparison tasks.

To ensure proper management and maniulation of data, uploaded images are validated for supported formats (JPEG, PNG) via error handling.

### Testing
The tests implemented in test_main.py primarily serve as unit tests for the FastAPI application. Each test function (e.g., `test_create_profile`, `test_update_profile`) verifies a specific endpoint or functionality of the application.