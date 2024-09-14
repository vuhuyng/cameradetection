import cv2
import numpy as np
import os
import io
from PIL import Image
import pickle

FACE_DIM = (50, 50)  # Define a consistent face dimension


def collect_data_from_camera():
    # Initialize camera
    camera = cv2.VideoCapture(0)
    face_data = []
    facecascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a window to display the video feed
    cv2.namedWindow("Collecting Face Data", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, FACE_DIM)
            face_data.append(face.flatten())  # Flatten the face data

        # Show the frame
        cv2.imshow("Collecting Face Data", frame)

        # Stop if ESC key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    # Release resources and close windows
    camera.release()
    cv2.destroyAllWindows()

    return np.array(face_data)


def save_face_data(faces, labels):
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    names_file_path = os.path.join(os.getcwd(), 'names.pkl')

    # Load existing data if available
    if os.path.exists(faces_file_path):
        with open(faces_file_path, 'rb') as f:
            existing_faces = pickle.load(f)
    else:
        existing_faces = np.empty((0, faces.shape[1]))

    if os.path.exists(names_file_path):
        with open(names_file_path, 'rb') as f:
            existing_labels = pickle.load(f)
    else:
        existing_labels = []

    # Append new data
    faces = np.vstack([existing_faces, faces])
    labels += existing_labels

    # Save updated data
    with open(faces_file_path, 'wb') as f:
        pickle.dump(faces, f)

    with open(names_file_path, 'wb') as f:
        pickle.dump(labels, f)


def load_data_from_images(image_stream):
    image = Image.open(image_stream).convert('RGB')
    image_array = np.array(image)

    # Assume face detection is done here and we have faces extracted
    # For demonstration, resizing face data to a fixed size
    face_data = []
    # Placeholder for actual face detection and processing
    # Example for resizing face data
    face = cv2.resize(image_array, (50, 50))  # Adjust the size if needed
    face_data.append(face.flatten())

    face_data = np.array(face_data)
    # Example name, replace with actual name if available
    names = ['DummyName']
    return face_data, names
