import cv2
import requests


SERVER_URL = "http://localhost:8000"

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Camera capture failed")
    
cv2.imwrite("capture.jpg", frame)

person_id = input("Enter person id: ")

url = f"{SERVER_URL}/enroll/{person_id}"

with open("capture.jpg", "rb") as f:
    response = requests.post(
        url,
        files={"file": f}
    )

# print(response.json())
print("Status code:", response.status_code)
print("Response text:")
print(response.text)