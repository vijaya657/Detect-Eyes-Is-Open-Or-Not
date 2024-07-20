import cv2

# Load haarcascade for eye detection from a local path
eye_cascPath = r'C:\Users\vijay\OneDrive\Desktop\Eye open or not\Closed-Eye-Detection-with-opencv\haarcascade_eye_tree_eyeglasses.xml'  # Update the path
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the aspect ratio of the eye region
        aspect_ratio = float(w) / h

        # Define thresholds for aspect ratio to determine eye state
        open_eye_ratio = 0.25  # Adjust this threshold as needed

        if aspect_ratio > open_eye_ratio:
            cv2.putText(frame, 'Eyes Open', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Eyes Closed', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Eye Detection', frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'):
        break

cap.release()
cv2.destroyAllWindows()
