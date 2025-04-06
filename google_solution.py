import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import tempfile
import os
import mediapipe as mp
import hashlib
import csv
from pathlib import Path

# Streamlit config
st.set_page_config(page_title="Voter Verification System", layout="centered")
st.title("🕳️ Voter Verification using Face Recognition")
st.markdown("This application verifies a voter by comparing the face on their ID card with a live camera capture.")

# Sidebar
st.sidebar.header("📷 Image Input Options")
use_camera = st.sidebar.checkbox("Use Camera", value=True)

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection

# Directory to store verified faces
verified_faces_dir = "verified_faces"
Path(verified_faces_dir).mkdir(exist_ok=True)

# Detect and crop face with padding and annotation
def detect_and_crop_face(image, label=""):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as face_detection:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_image)

        if not results.detections:
            st.warning(f"No face detected in {label} image.")
            return None, image

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = image.shape
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        # Add padding
        padding = 20
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, w)
        y2 = min(y2 + padding, h)

        cropped_face = image[y1:y2, x1:x2]

        # Annotate image
        annotated = image.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return cropped_face, annotated

# Compare with existing verified faces using DeepFace
def is_duplicate_deepface(new_face_path, folder=verified_faces_dir):
    for fname in os.listdir(folder):
        existing_path = os.path.join(folder, fname)
        try:
            result = DeepFace.verify(
                img1_path=new_face_path,
                img2_path=existing_path,
                model_name="ArcFace",
                enforce_detection=False
            )
            if result.get("verified", False):
                return True
        except:
            continue
    return False

# Save verified face image
face_id_counter = Path(verified_faces_dir) / "counter.txt"
if not face_id_counter.exists():
    face_id_counter.write_text("0")

def save_verified_face_image(image):
    count = int(face_id_counter.read_text()) + 1
    face_id_counter.write_text(str(count))
    filename = f"face_{count}.jpg"
    path = os.path.join(verified_faces_dir, filename)
    cv2.imwrite(path, image)

# Image input
def capture_image(label):
    st.subheader(f"{label} Image")

    if use_camera:
        picture = st.camera_input(f"Capture {label}")
        if picture:
            img = Image.open(picture)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return img
    else:
        uploaded_file = st.file_uploader(f"Upload {label} image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            img = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            return img

    return None

# Load images
id_image = capture_image("ID Card")
live_image = capture_image("Live Face")

# Main button
if st.button("🔍 Verify Voter Identity"):
    if id_image is None or live_image is None:
        st.warning("Please provide both images before verification.")
    else:
        with st.spinner("Detecting faces and comparing..."):
            try:
                # Detect and crop
                id_face, id_annotated = detect_and_crop_face(id_image, "ID Card")
                live_face, live_annotated = detect_and_crop_face(live_image, "Live Face")

                if id_face is None or live_face is None:
                    st.error("Face not detected in one of the images.")
                else:
                    # Resize to standard size
                    target_size = (224, 224)
                    id_face_resized = cv2.resize(id_face, target_size)
                    live_face_resized = cv2.resize(live_face, target_size)

                    # Show annotated faces
                    st.image(cv2.cvtColor(id_annotated, cv2.COLOR_BGR2RGB), caption="ID Card Face", use_container_width=True)
                    st.image(cv2.cvtColor(live_annotated, cv2.COLOR_BGR2RGB), caption="Live Face", use_container_width=True)

                    # Save temp images for comparison
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1, \
                         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
                        cv2.imwrite(tmp1.name, id_face_resized)
                        cv2.imwrite(tmp2.name, live_face_resized)

                        result = DeepFace.verify(
                            img1_path=tmp1.name,
                            img2_path=tmp2.name,
                            model_name="ArcFace",
                            enforce_detection=False
                        )

                    verified = result.get("verified", False)
                    distance = result.get("distance", None)
                    threshold = result.get("threshold", None)

                    if verified:
                        # Check for duplicate using deepface
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as new_tmp:
                            cv2.imwrite(new_tmp.name, live_face_resized)
                            if is_duplicate_deepface(new_tmp.name):
                                st.warning("⚠️ Duplicate vote detected. This person has already voted.")
                            else:
                                save_verified_face_image(live_face_resized)
                                st.success(f"✅ Voter verified!\nDistance: {distance:.4f} (Threshold: {threshold:.4f})")
                                st.balloons()
                        os.remove(new_tmp.name)
                    else:
                        st.error(f"❌ Face mismatch.\nDistance: {distance:.4f} (Threshold: {threshold:.4f})")

                    # Clean up temp files
                    os.remove(tmp1.name)
                    os.remove(tmp2.name)

            except Exception as e:
                st.error(f"Verification failed: {e}")
