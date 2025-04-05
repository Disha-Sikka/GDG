import cv2
import os
import uuid
import face_recognition

# Create dataset folder if it doesn't exist
dataset_path = "voter_faces"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

def get_face_encoding(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_image)
    if len(boxes) == 0:
        return None
    encoding = face_recognition.face_encodings(rgb_image, boxes)[0]
    return encoding

# === Step 1: Scan Voter ID Card ===
cap = cv2.VideoCapture('http://192.168.1.6:8080/video')
print("[INFO] Hold your voter ID card in front of the webcam...")
print("Press Enter to scan the voter ID card.")

id_face = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Couldn't access camera.")
        break

    # Display resized frame for preview
    display_frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Scan Voter ID Card", display_frame)

    if cv2.waitKey(1) == 13:  # Press Enter
        id_face = frame.copy()
        break

if id_face is None:
    print("[ERROR] No frame captured for ID card.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# === Step 2: Scan Live Face ===
print("[INFO] Now scanning live face... Press Enter to capture your face.")

live_face = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Couldn't access camera.")
        break

    # Display resized frame for preview
    display_frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Live Face", display_frame)

    if cv2.waitKey(1) == 13:  # Press Enter
        live_face = frame.copy()
        break

# === Step 3: Compare Faces ===
if live_face is not None:
    try:
        id_encoding = get_face_encoding(id_face)
        live_encoding = get_face_encoding(live_face)

        if id_encoding is None:
            print("[❌] No face detected on ID card. Try again.")
        elif live_encoding is None:
            print("[❌] No face detected in live camera. Try again.")
        else:
            result = face_recognition.compare_faces([id_encoding], live_encoding)
            if result[0]:
                print("\n✅ Voter verified successfully!")

                # Save the image
                voter_id = str(uuid.uuid4())
                filename = os.path.join(dataset_path, f"{voter_id}.jpg")
                cv2.imwrite(filename, cv2.cvtColor(live_face, cv2.COLOR_BGR2RGB))
                print(f"[INFO] Face saved to {filename}")
            else:
                print("\n❌ Face does not match the voter ID card.")
    except Exception as e:
        print(f"[ERROR] Failed to encode or compare faces: {str(e)}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
