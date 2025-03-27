from capture_face import capture_voter_image
from verify_voter import verify_voter

# Step 1: Enter voter name
voter_name = input("Enter Voter's Name: ")
voter_age = input("Enter voter age: ")

# Step 2: Capture the voter's face
capture_voter_image(voter_name,voter_age)

# Step 3: Verify if the voter is registered
if verify_voter(voter_name):
    print("Voter verified! Proceed to vote.")
else:
    print("Voter NOT registered!")

