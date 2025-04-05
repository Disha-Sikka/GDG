# face recognize
import cv2
import os
import face_recognition

dataset_path = "voter_faces"
voted_voters = set()  # Store voter names who already voted

# Load known voter faces
known_faces = {}
for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)
    
    if len(encoding) > 0:
        known_faces[file.split(".")[0]] = encoding[0]  # Store as {Name: Encoding}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturing image.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(list(known_faces.values()), encoding)

        if True in matches:
            matched_idx = matches.index(True)
            voter_name = list(known_faces.keys())[matched_idx]

            if voter_name in voted_voters:
                text = f"{voter_name} - Already Voted!"
                color = (0, 0, 255)  # Red for rejected
            else:
                text = f"{voter_name} - Verified!"
                voted_voters.add(voter_name)  # Mark as voted
                color = (0, 255, 0)  # Green for verified

            # Draw a box and label
            (top, right, bottom, left) = location
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Voter Verification", frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
