# Voter Verification System using Face Recognition

This is a Streamlit-based web application that verifies a voter's identity by comparing the face on their ID card with a live captured face. It also prevents duplicate voting by saving previously verified faces and checking against them.

## Features

- Face detection using MediaPipe
- Face comparison using DeepFace (ArcFace or Facenet model)
- Prevents duplicate votes using stored facial data
- Upload or capture ID card and live face
- Option to clear saved voter data

## Requirements

Install the following Python packages using the provided `requirements.txt`:

```
tf-keras
mediapipe
tensorflow
pytesseract
streamlit
opencv-python
Pillow
numpy
deepface


## How It Works

- User uploads or captures two images: one from the ID card and one live face.
- The app detects and crops the faces from both images.
- The faces are compared using DeepFace.
- If the faces match, the system checks whether this person has already voted.
- If no match is found in previous records, their face is saved and vote is considered valid.

## Preventing Duplicate Voting

The app saves every verified face in a local folder called `verified_faces`. On each new attempt, it compares the new face with all saved ones using DeepFace. If a match is found, the system shows a "Duplicate vote" warning.

To reset and clear all stored data, use the “Clear Verified Data” button in the sidebar.

