import cv2
import numpy as np
import dlib
from imutils import face_utils
import simpleaudio as sa  # Import simpleaudio

# Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initializing the face detector and landmark d-----------------------------------------------------------------------------------------------etector
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
except RuntimeError as e:
    print(f"Error loading shape predictor: {e}")
    exit()

# Load the sound file (update with your audio file path)
wave_obj = sa.WaveObject.from_wave_file("music.wav")  # Specify your sound file path here

# Status marking for current state
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
is_playing = False  # Flag to track if the sound is playing


def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist


def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Frame could not be captured.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if len(faces) == 0:
        print("No faces detected.")
        continue  # Skip to the next iteration if no faces are detected

    # Initialize face_frame to the current frame
    face_frame = frame.copy()

    # Detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Now judge what to do for the eye blinks
        if left_blink == 0 or right_blink == 0:
            drowsy += 1
            active = 0  # Reset active state
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                if not is_playing:  # Check if the sound is not already playing
                    wave_obj.play()  # Play the sound when DROWSY status is detected
                    is_playing = True  # Set the flag to True
        else:
            drowsy = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                if is_playing:  # Check if the sound is currently playing
                    is_playing = False  # Set the flag to False
                    sa.stop_all()  # Stop all sounds

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
