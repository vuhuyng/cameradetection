import cv2
import numpy as np
import pickle
import os
from PIL import Image


def collect_data_from_camera():
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    data = []
    collected_data = False

    while not collected_data:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50)).flatten()
            data.append(face)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Collecting Face Data', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            collected_data = True

    cap.release()
    cv2.destroyAllWindows()

    # Save collected data
    if len(data) > 0:
        faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
        names_file_path = os.path.join(os.getcwd(), 'names.pkl')

        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                faces = pickle.load(f)
            faces = np.vstack((faces, np.array(data)))
        else:
            faces = np.array(data)

        with open(faces_file_path, 'wb') as f:
            pickle.dump(faces, f)

        st.success("Data collected successfully!")
    else:
        st.warning("No data collected.")


def save_face_data(faces, labels):
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    names_file_path = os.path.join(os.getcwd(), 'names.pkl')

    with open(faces_file_path, 'wb') as f:
        pickle.dump(faces, f)
    with open(names_file_path, 'wb') as f:
        pickle.dump(labels, f)


def load_data_from_images(images):
    faces = []
    labels = []
    for img in images:
        img = np.array(img)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, (50, 50)).flatten()
            faces.append(face)
            labels.append("Unknown")

    if len(faces) > 0:
        faces = np.array(faces)
        labels = np.array(labels)
    else:
        faces = np.empty((0, 2500))
        labels = []

    return faces, labels
