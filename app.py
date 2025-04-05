import streamlit as st
import cv2
import os
import numpy as np
import face_recognition
from PIL import Image
import tempfile

st.set_page_config(page_title="Voter Verification", layout="centered")

st.title("🗳️ Voter Verification System")

dataset_path = "voter_faces"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

def capture_image_from_webcam(label):
    st.write(f"📸 Capture: {label}")
    run = st.button("Start Camera" if label == "Live Face" else "Scan ID Card")
    if run:
        cap = cv2.VideoCapture(0)
        captured_image = None
        frame_placeholder = st.empty()
        capture_button = st.button("📷 Capture Image")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the camera.")
                break
            
            frame_resized = cv2.resize(frame, (480, 320))  # Resize for performance
            frame_placeholder.image(frame_resized, channels="BGR")

            if capture_button:
                captured_image = frame
                break

        cap.release()
        return captured_image
    return None

def extract_face_from_image(image_np):
    face_locations = face_recognition.face_locations(image_np)
    if not face_locations:
        return None
    top, right, bottom, left = face_locations[0]
    face_image = image_np[top:bottom, left:right]
    return face_image

def verify_faces(face1_np, face2_np):
    try:
        face1_encoding = face_recognition.face_encodings(face1_np)[0]
        face2_encoding = face_recognition.face_encodings(face2_np)[0]
        result = face_recognition.compare_faces([face1_encoding], face2_encoding)
        return result[0]
    except IndexError:
        return False

st.subheader("Step 1: Scan Voter ID Card")
id_card_image = capture_image_from_webcam("ID Card")

if id_card_image is not None:
    id_face = extract_face_from_image(id_card_image)
    if id_face is not None:
        st.image(id_face, caption="Detected Face from ID Card", width=200)
    else:
        st.error("No face detected on ID card. Please try again.")

st.subheader("Step 2: Capture Live Face")
live_image = capture_image_from_webcam("Live Face")

if live_image is not None and id_card_image is not None:
    live_face = extract_face_from_image(live_image)

    if live_face is not None:
        st.image(live_face, caption="Captured Live Face", width=200)

        st.subheader("Step 3: Verifying...")
        if verify_faces(id_face, live_face):
            st.success("✅ Face Match Successful! You can now vote.")
            voter_name = st.text_input("Enter Voter's Name to Save Record:")
            if st.button("Save Voter Face"):
                save_path = os.path.join(dataset_path, voter_name + ".jpg")
                cv2.imwrite(save_path, live_face)
                st.info(f"Voter face saved as {voter_name}.jpg")
        else:
            st.error("❌ Face Mismatch! Verification Failed.")
    else:
        st.error("No face detected from live capture.")
