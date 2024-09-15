import streamlit as st
import cv2
import numpy as np
from PIL import Image
from algorithm import train_knn
from collection_data import collect_data_from_camera, save_face_data, load_data_from_images
import pickle
import os
import io
import pandas as pd
import datetime

# Streamlit app functions

def check_login(username, password):
    return username == 'admin' and password == '12'

def load_data():
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    names_file_path = os.path.join(os.getcwd(), 'names.pkl')

    faces = np.empty((0, 2500))  # Adjust size if needed
    labels = []

    try:
        if os.path.exists(faces_file_path):
            with open(faces_file_path, 'rb') as f:
                faces = pickle.load(f)
        if os.path.exists(names_file_path):
            with open(names_file_path, 'rb') as f:
                labels = pickle.load(f)
    except (IOError, pickle.UnpicklingError) as e:
        st.error(f"Failed to load data: {e}")

    return faces, labels

def save_face_data(faces, labels):
    faces_file_path = os.path.join(os.getcwd(), 'faces.pkl')
    names_file_path = os.path.join(os.getcwd(), 'names.pkl')

    with open(faces_file_path, 'wb') as f:
        pickle.dump(faces, f)
    with open(names_file_path, 'wb') as f:
        pickle.dump(labels, f)

def save_attendance_to_csv(attendance_data):
    df = pd.DataFrame(attendance_data, columns=['Name', 'Timestamp'])
    csv_file_path = os.path.join(os.getcwd(), 'attendance.csv')

    try:
        if os.path.exists(csv_file_path):
            existing_df = pd.read_csv(csv_file_path)
            combined_df = pd.concat([existing_df, df])
            combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
            combined_df = combined_df.sort_values(by='Timestamp')
            combined_df = combined_df.drop_duplicates(
                subset='Name', keep='first')
        else:
            combined_df = df

        combined_df.to_csv(csv_file_path, index=False)
        st.success("Attendance data saved successfully!")
    except IOError as e:
        st.error(f"Failed to save attendance: {e}")

def draw_ai_tech_bbox(frame, x, y, w, h, color=(0, 255, 0), thickness=2):
    corner_length = 30
    line_thickness = thickness
    glow_color = (0, 255, 0)  # Neon cyan-like glow

    cv2.line(frame, (x, y), (x + corner_length, y), glow_color, line_thickness)
    cv2.line(frame, (x, y), (x, y + corner_length), glow_color, line_thickness)
    cv2.line(frame, (x + w, y), (x + w - corner_length, y), glow_color, line_thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length), glow_color, line_thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length), glow_color, line_thickness)
    cv2.line(frame, (x, y + h), (x + corner_length, y + h), glow_color, line_thickness)
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), glow_color, line_thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), glow_color, line_thickness)

    cv2.line(frame, (x, y + h // 2), (x + w, y + h // 2), glow_color, 1)
    cv2.line(frame, (x + w // 2, y), (x + w // 2, y + h), glow_color, 1)

    cv2.putText(frame, "SCANNING...", (x - 200, y + 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), glow_color, -1)
    alpha = 0.1
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "ID: FPT UNIVERSITY", (x + w + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, glow_color, 1)
    cv2.putText(frame, "Status: Processing", (x + w + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, glow_color, 1)

def display_face_recognition():
    st.header("Realtime Face Recognition")

    faces, labels = load_data()

    if faces.size == 0 or len(labels) == 0:
        st.error("No data available for training!")
        return

    if faces.shape[0] < 1:
        st.error("Insufficient training data!")
        return

    knn = train_knn(faces, labels)
    stframe = st.empty()

    detected_names = {}
    attendance_data = []

    # Attempt to open the camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Cannot access the camera! Please check if your camera is connected and try again.")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Cannot read from the camera! Please check if your camera is working correctly.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_coordinates = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(face_coordinates) == 0:
            st.write("No faces detected.")
        else:
            st.write(f"Faces detected: {face_coordinates}")

        for (x, y, w, h) in face_coordinates:
            face = frame[y:y + h, x:x + w, :]
            resized_face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

            if faces.shape[0] > 0:
                name = knn.predict(resized_face)[0]

                bbox_color = (0, 255, 255) if name not in detected_names else (255, 0, 255)
                draw_ai_tech_bbox(frame, x, y, w, h, bbox_color, 3)

                label = f"{name} - {'New' if name not in detected_names else 'Checked'}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if name not in detected_names:
                    # Save attendance data immediately and update last saved timestamp
                    attendance_data.append([name, timestamp])
                    detected_names[name] = timestamp
                    save_attendance_to_csv(attendance_data)
                else:
                    # Check if the time since last saved entry is more than a threshold
                    last_saved_timestamp = datetime.datetime.strptime(detected_names[name], "%Y-%m-%d %H:%M:%S")
                    current_timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                    if (current_timestamp - last_saved_timestamp).seconds > 60:  # 1 minute threshold
                        # Update timestamp for existing detection
                        attendance_data = [entry for entry in attendance_data if entry[0] != name]
                        attendance_data.append([name, timestamp])
                        detected_names[name] = timestamp
                        # Re-save with updated timestamps
                        save_attendance_to_csv(attendance_data)

        stframe.image(frame, channels="BGR")

        # Break the loop if the user presses the 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()


    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Cannot read from the camera!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_coordinates = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(face_coordinates) == 0:
            st.write("No faces detected.")
        else:
            st.write(f"Faces detected: {face_coordinates}")

        for (x, y, w, h) in face_coordinates:
            face = frame[y:y + h, x:x + w, :]
            resized_face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

            if faces.shape[0] > 0:
                name = knn.predict(resized_face)[0]

                bbox_color = (0, 255, 255) if name not in detected_names else (255, 0, 255)
                draw_ai_tech_bbox(frame, x, y, w, h, bbox_color, 3)

                label = f"{name} - {'New' if name not in detected_names else 'Checked'}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if name not in detected_names:
                    # Save attendance data immediately and update last saved timestamp
                    attendance_data.append([name, timestamp])
                    detected_names[name] = timestamp
                    save_attendance_to_csv(attendance_data)
                else:
                    # Check if the time since last saved entry is more than a threshold
                    last_saved_timestamp = datetime.datetime.strptime(detected_names[name], "%Y-%m-%d %H:%M:%S")
                    current_timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

                    if (current_timestamp - last_saved_timestamp).seconds > 60:  # 1 minute threshold
                        # Update timestamp for existing detection
                        attendance_data = [entry for entry in attendance_data if entry[0] != name]
                        attendance_data.append([name, timestamp])
                        detected_names[name] = timestamp
                        # Re-save with updated timestamps
                        save_attendance_to_csv(attendance_data)

        stframe.image(frame, channels="BGR")

        # Break the loop if the user presses the 'ESC' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()

def view_attendance():
    st.header("View Attendance")

    csv_file_path = os.path.join(os.getcwd(), 'attendance.csv')
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        st.dataframe(df)

        # Provide download option
        st.download_button(
            label="Download Attendance CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='attendance.csv',
            mime='text/csv'
        )
    else:
        st.warning("No attendance data available!")

def admin_mode():
    st.header("Admin Panel")

    action = st.selectbox("Choose an action:", [
        "Collect Data from Camera", "Upload Multiple Images", "Export Attendance", "View Attendance"])

    if action == "Collect Data from Camera":
        name = st.text_input("Enter your name:")
        if st.button("Start Data Collection"):
            if name:
                st.write("Collecting data from camera...")
                collect_data_from_camera(name)
                st.write("Data collection complete!")
                save_face_data(*load_data_from_images())  # Save collected data
            else:
                st.warning("Please enter your name.")

    elif action == "Upload Multiple Images":
        uploaded_files = st.file_uploader("Choose image files", accept_multiple_files=True)
        if st.button("Upload Images"):
            if uploaded_files:
                for file in uploaded_files:
                    image = Image.open(file)
                    img_array = np.array(image)
                    collect_data_from_camera(img_array, file.name)  # Use a placeholder function to handle this
                st.write("Images uploaded and processed successfully!")
                save_face_data(*load_data_from_images())  # Save uploaded images
            else:
                st.warning("Please upload at least one image.")

    elif action == "Export Attendance":
        save_attendance_to_csv(load_data_from_images())  # Save data as CSV
        st.write("Attendance data exported successfully!")

    elif action == "View Attendance":
        view_attendance()

def main():
    st.title("Face Recognition System")

    mode = st.sidebar.selectbox("Select Mode", ["User", "Admin"])

    if mode == "User":
        st.header("Face Recognition")

        if st.button("Start"):
            display_face_recognition()

    elif mode == "Admin":
        st.subheader("Admin Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.success("Login successful!")
                admin_mode()
            else:
                st.error("Invalid username or password")

if __name__ == "__main__":
    main()
