import cv2
import os

dataset_path = "voter_faces"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

def capture_voter_image(name):
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
            cv2.imwrite(voter_image_path, frame)
            print(f"Voter {name}'s face saved!")
            break

    cap.release()
    cv2.destroyAllWindows()
