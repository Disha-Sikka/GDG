import os

dataset_path = "voter_faces"

def verify_voter(name):
    voter_image_path = os.path.join(dataset_path, name + ".jpg")
    
    if os.path.exists(voter_image_path):
        print("Voter is registered! Proceed to voting.")
        return True
    else:
        print("Voter is NOT registered!")
        return False
