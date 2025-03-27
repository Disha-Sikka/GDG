import cv2
import os
import csv
import face_recognition
import numpy as np

dataset_path = "voter_faces"

# Ensure the dataset directory exists
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# CSV File to store voter details
csv_file = "voters.csv"

# Ensure the CSV file exists
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Age", "Image_Path"])  # Header

# Function to check if face already exists
def is_face_duplicate(new_face_encoding):
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                image_path = row[2]
                if os.path.exists(image_path):
                    known_image = face_recognition.load_image_file(image_path)
                    known_encoding = face_recognition.face_encodings(known_image)
                    if known_encoding:
                        known_encoding = known_encoding[0]
                        # Compare the new face with existing one
                        results = face_recognition.compare_faces([known_encoding], new_face_encoding)
                        if results[0]:  # If match found
                            return True
    except Exception as e:
        print(f"Error checking face: {e}")
    return False

# Function to capture voter image
def capture_voter_image(name, age):
    cap = cv2.VideoCapture(0)
    voter_image_path = os.path.join(dataset_path, name + ".jpg")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing image.")
            break

        cv2.imshow("Capture Voter Face", frame)

        # Press 's' to save the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Convert frame to face encoding
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_frame)

            if face_encodings:
                if is_face_duplicate(face_encodings[0]):
                    print("Error: Vote already casted!")
                else:
                    cv2.imwrite(voter_image_path, frame)
                    print(f"Image saved as {voter_image_path}")
                    
                    # Save voter details
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([name, age, voter_image_path])
                    print("Voter details saved.")
            else:
                print("No face detected! Try again.")
            break

    cap.release()
    cv2.destroyAllWindows()

