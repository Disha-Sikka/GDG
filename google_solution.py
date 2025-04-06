import streamlit as st
try:
    import cv2
    st.success("✅ OpenCV (cv2) imported successfully!")
except ModuleNotFoundError:
    st.error("❌ OpenCV (cv2) not installed!")

import streamlit as st
import cv2
import pytesseract
import face_recognition
from PIL import Image
import numpy as np
import pandas as pd
import os
import uuid

# ---------- CSV Setup ----------
CSV_FILE = "voted_voters.csv"
VOTER_RECORDS_FILE = "voter_records.csv"
dataset_path = "voter_faces"

def init_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=["voter_id"])
        df.to_csv(CSV_FILE, index=False)
    if not os.path.exists(VOTER_RECORDS_FILE):
        df = pd.DataFrame(columns=["voter_id", "ocr_text", "card_face", "live_face"])
        df.to_csv(VOTER_RECORDS_FILE, index=False)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

def has_already_voted(voter_id):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        return voter_id in df["voter_id"].values
    return False

def mark_voter_as_voted(voter_id):
    df = pd.read_csv(CSV_FILE)
    df = pd.concat([df, pd.DataFrame([{"voter_id": voter_id}])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

def save_voter_record(voter_id, ocr_text, card_face_path, live_face_path):
    df = pd.read_csv(VOTER_RECORDS_FILE)
    df = pd.concat([df, pd.DataFrame([{
        "voter_id": voter_id,
        "ocr_text": ocr_text,
        "card_face_path": card_face_path,
        "live_face_path": live_face_path
    }])], ignore_index=True)
    df.to_csv(VOTER_RECORDS_FILE, index=False)

def extract_voter_id(text):
    lines = text.split('\n')
    for line in lines:
        if any(char.isdigit() for char in line) and any(char.isalpha() for char in line):
            return line.strip()
    return None

# ---------- Streamlit App ----------
st.set_page_config(page_title="Voter Verification System", layout="centered")
st.title("🗳️ Secure Voter Verification System")
init_csv()
ip = st.sidebar.text_input("Enter your webcam IP address", "http://192.168.0.100:8080/video")

# Session state setup
for key in ["verification_status", "ocr_text", "voter_id", "voter_id_image", "live_face_image", "card_face_path"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------- Functions ----------
def image_to_bytes(img):
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def extract_text_and_boxes(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    boxes = []
    extracted_text = ""
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            boxes.append((x, y, w, h, data['text'][i]))
            extracted_text += data['text'][i] + " "
    return extracted_text, boxes

def draw_boxes(image, boxes):
    for (x, y, w, h, word) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, word, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def is_voter_card(text):
    keywords = ["ELECTION", "COMMISSION", "INDIA", "VOTER", "IDENTITY", "CARD"]
    count = sum(1 for word in keywords if word in text.upper())
    return count >= 3

def extract_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    if faces:
        top, right, bottom, left = faces[0]
        return rgb[top:bottom, left:right]
    return None

def capture_from_ip_camera(label):
    st.write(f"📸 Capture: {label}")
    # Try converting to int if it's a number (like '0' or '1'), else keep as string
    try:
        source = int(ip)
    except ValueError:
        source = ip

    cap = cv2.VideoCapture(source)

    frame_placeholder = st.empty()
    capture_button = st.button(f"📷 Capture Frame for {label}")

    img = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Cannot access the camera stream.")
            break
        frame_resized = cv2.resize(frame, (480, 320))
        frame_placeholder.image(frame_resized, channels="BGR")
        if capture_button:
            img = frame
            frame_placeholder.image(img, caption=f"Captured {label}", channels="BGR")
            cap.release()
            break
    return img

def get_face_encoding(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_img)
    if boxes:
        return face_recognition.face_encodings(rgb_img, known_face_locations=boxes)[0]
    return None

def is_face_already_voted(live_encoding, tolerance=0.5):
    for filename in os.listdir(dataset_path):
        if filename.endswith("_live.jpg"):
            existing_img = face_recognition.load_image_file(os.path.join(dataset_path, filename))
            existing_encs = face_recognition.face_encodings(existing_img)
            if existing_encs:
                match = face_recognition.compare_faces([existing_encs[0]], live_encoding, tolerance=tolerance)
                if match[0]:
                    return True
    return False

# ---------- Menu ----------
menu = ["🏠 Home", "🦔 Scan Voter Card", "🧑 Live Face Capture", "🔍 Verify", "✅ Vote"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "🦔 Scan Voter Card":
    frame = capture_from_ip_camera("Voter Card")
    if frame is not None:
        st.image(frame, caption="Captured Voter Card", channels="BGR")
        st.session_state.voter_id_image = frame

        text, boxes = extract_text_and_boxes(frame)
        boxed_image = draw_boxes(frame.copy(), boxes)
        st.image(boxed_image, caption="Detected Text with Boxes", channels="BGR")
        st.write("### Extracted Text:")
        st.write(text)

        if is_voter_card(text):
            voter_id = extract_voter_id(text)
            if not voter_id:
                st.error("❌ Voter ID not found in text.")
            elif has_already_voted(voter_id):
                st.warning("⚠️ This voter has already cast their vote.")
            else:
                st.success(f"✅ Voter ID Verified: {voter_id}")
                st.session_state.voter_id = voter_id
                st.session_state.ocr_text = text

                card_face = extract_face(frame)
                if card_face is not None:
                    card_face_path = os.path.join(dataset_path, f"{voter_id}_card.jpg")
                    cv2.imwrite(card_face_path, cv2.cvtColor(card_face, cv2.COLOR_RGB2BGR))
                    st.session_state.card_face_path = card_face_path
                    st.image(card_face, caption="Extracted Face from Card", channels="RGB")
                else:
                    st.warning("⚠️ No face detected on Voter Card.")
        else:
            st.error("❌ Not verified as a Voter Card.")

elif choice == "🧑 Live Face Capture":
    face_img = capture_from_ip_camera("Live Face")
    if face_img is not None:
        st.session_state.live_face_image = face_img

elif choice == "🔍 Verify":
    if st.session_state.voter_id_image is None or st.session_state.live_face_image is None:
        st.warning("Please upload your Voter ID and capture your live face first.")
    else:
        st.subheader("🔎 Verifying...")

        id_encoding = get_face_encoding(st.session_state.voter_id_image)
        live_encoding = get_face_encoding(st.session_state.live_face_image)

        if id_encoding is not None and live_encoding is not None:
            # Load already voted encodings
            duplicate_found = False
            if os.path.exists(CSV_FILE):
                df = pd.read_csv(CSV_FILE)
                for _, row in df.iterrows():
                    if "live_encoding" in row and pd.notna(row["live_encoding"]):
                        known_encoding = np.fromstring(row["live_encoding"], sep=',')
                        match = face_recognition.compare_faces([known_encoding], live_encoding)[0]
                        if match:
                            duplicate_found = True
                            break

            if duplicate_found:
                st.error("❌ This face has already been used to vote.")
                st.session_state.verification_status = False
            elif has_already_voted(st.session_state.voter_id):
                st.error("⚠️ This voter ID has already been used to vote.")
                st.session_state.verification_status = False
            else:
                st.success("✅ Face Matched. Verification Successful!")
                st.session_state.verification_status = True
                st.session_state.live_encoding = live_encoding  # Save for later write
        else:
            st.error("⚠️ Face not detected in one or both images.")

elif choice == "✅ Vote":
    if not st.session_state.verification_status:
        st.warning("Please complete verification before voting.")
    else:
        st.subheader("🗳️ Cast Your Vote")
        party = st.radio("Choose your party:", ["Party A", "Party B", "Party C"])
        if st.button("Submit Vote"):
            # Save live face image
            live_face = extract_face(st.session_state.live_face_image)
            if live_face is not None:
                live_face_path = os.path.join(dataset_path, f"{st.session_state.voter_id}_live.jpg")
                cv2.imwrite(live_face_path, cv2.cvtColor(live_face, cv2.COLOR_RGB2BGR))
            else:
                live_face_path = None
                st.warning("⚠️ No face detected in live image.")

            # Save to voter record
            save_voter_record(
                st.session_state.voter_id,
                st.session_state.ocr_text,
                st.session_state.card_face_path,
                live_face_path
            )

            # Save to voted CSV with live encoding
            if st.session_state.live_encoding is not None:
                df = pd.read_csv(CSV_FILE)
                df = pd.concat([df, pd.DataFrame([{
                    "voter_id": st.session_state.voter_id,
                    "live_encoding": ','.join(map(str, st.session_state.live_encoding))
                }])], ignore_index=True)
                df.to_csv(CSV_FILE, index=False)

            st.success(f"🎉 Your vote for {party} has been cast successfully!")
            # Reset status to prevent repeat voting without re-verification
            st.session_state.verification_status = False
