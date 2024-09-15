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
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


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
    cv2.line(frame, (x + w, y), (x + w - corner_length, y),
             glow_color, line_thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_length),
             glow_color, line_thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_length),
             glow_color, line_thickness)
    cv2.line(frame, (x, y + h), (x + corner_length, y + h),
             glow_color, line_thickness)
    cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h),
             glow_color, line_thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h -
             corner_length), glow_color, line_thickness)

    cv2.line(frame, (x, y + h // 2), (x + w, y + h // 2), glow_color, 1)
    cv2.line(frame, (x + w // 2, y), (x + w // 2, y + h), glow_color, 1)

    cv2.putText(frame, "SCANNING...", (x - 200, y + 100),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), glow_color, -1)
    alpha = 0.1
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "ID: FPT UNIVERSITY", (x + w + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, glow_color, 1)
    cv2.putText(frame, "Status: Processing", (x + w + 10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, glow_color, 1)


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

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


class FaceDetectionTransformer(VideoTransformerBase):
    def __init__(self, knn, faces, labels, detected_names, attendance_data):
        self.knn = knn
        self.faces = faces
        self.labels = labels
        self.detected_names = detected_names
        self.attendance_data = attendance_data

    def transform(self, frame):
        # Chuyển đổi frame thành mảng NumPy
        frame = frame.to_ndarray(format="bgr24")

        # Xử lý nhận diện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facecascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in face_coordinates:
            face = frame[y:y + h, x:x + w, :]
            resized_face = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

            if self.faces.shape[0] > 0:
                name = self.knn.predict(resized_face)[0]

                bbox_color = (0, 255, 255) if name not in self.detected_names else (
                    255, 0, 255)
                draw_ai_tech_bbox(frame, x, y, w, h, bbox_color, 3)

                label = f"{name} - {'New' if name not in self.detected_names else 'Checked'}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if name not in self.detected_names:
                    self.attendance_data.append([name, timestamp])
                    self.detected_names[name] = timestamp
                    save_attendance_to_csv(self.attendance_data)
                else:
                    last_saved_timestamp = datetime.datetime.strptime(
                        self.detected_names[name], "%Y-%m-%d %H:%M:%S")
                    current_timestamp = datetime.datetime.strptime(
                        timestamp, "%Y-%m-%d %H:%M:%S")

                    if (current_timestamp - last_saved_timestamp).seconds > 60:  # 1 minute threshold
                        self.attendance_data = [
                            entry for entry in self.attendance_data if entry[0] != name]
                        self.attendance_data.append([name, timestamp])
                        self.detected_names[name] = timestamp
                        save_attendance_to_csv(self.attendance_data)

        return frame


def display_face_recognition():
    st.header("Realtime Face Recognition")

    # Load dữ liệu khuôn mặt và nhãn
    faces, labels = load_data()

    if faces.size == 0 or len(labels) == 0:
        st.error("No data available for training!")
        return

    if faces.shape[0] < 1:
        st.error("Insufficient training data!")
        return

    knn = train_knn(faces, labels)
    detected_names = {}
    attendance_data = []

    # Sử dụng WebRTC để truy cập camera và xử lý video
    webrtc_streamer(
        key="face-recognition",
        video_transformer_factory=lambda: FaceDetectionTransformer(
            knn, faces, labels, detected_names, attendance_data)
    )


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
                face_data = collect_data_from_camera()

                if face_data.size == 0:
                    st.error("No face data collected!")
                    return

                faces, labels = load_data()

                if faces.size == 0:
                    faces = face_data
                    labels = [name] * face_data.shape[0]
                else:
                    if face_data.shape[1] == faces.shape[1]:
                        faces = np.vstack([faces, face_data])
                        labels += [name] * face_data.shape[0]
                    else:
                        st.error("Face data dimensions do not match!")
                        return
                save_face_data(faces, labels)
                st.success("Data collection completed and saved!")
            else:
                st.error("Please enter your name before starting data collection.")

    elif action == "Upload Multiple Images":
        uploaded_files = st.file_uploader(
            "Choose images", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])
        name = st.text_input("Enter your name:")

        if st.button("Upload Images"):
            if uploaded_files:
                # Initialize faces and labels
                faces, labels = load_data()

                if faces.size == 0:
                    # If no existing face data, initialize empty list
                    faces = np.empty((0, 2500))  # Adjust size if needed
                face_data_list = []

                for uploaded_file in uploaded_files:
                    image_data = io.BytesIO(uploaded_file.read())
                    face_data, _ = load_data_from_images(image_data)

                    # Ensure that face data dimensions match
                    if faces.size == 0 or face_data.shape[1] == faces.shape[1]:
                        face_data_list.append(face_data)
                    else:
                        st.error(
                            "Face data dimensions from uploaded images do not match!")
                        return

                if face_data_list:
                    face_data = np.vstack(face_data_list)

                    if name:
                        if face_data.size == 0:
                            st.error("No face data found in uploaded images!")
                            return

                        if faces.size == 0:
                            faces = face_data
                            labels = [name] * face_data.shape[0]
                        else:
                            if face_data.shape[1] == faces.shape[1]:
                                faces = np.vstack([faces, face_data])
                                labels += [name] * face_data.shape[0]
                            else:
                                st.error("Face data dimensions do not match!")
                                return
                        save_face_data(faces, labels)
                        st.success(
                            f"Successfully uploaded {len(uploaded_files)} images for {name}")
                    else:
                        st.error("Please enter your name.")
            else:
                st.error("Please upload some images.")

    elif action == "Export Attendance":
        csv_file_path = os.path.join(os.getcwd(), 'attendance.csv')
        if os.path.exists(csv_file_path):
            with open(csv_file_path, 'rb') as f:
                st.download_button(
                    label='Download Attendance CSV',
                    data=f.read(),
                    file_name='attendance.csv',
                    mime='text/csv'
                )
        else:
            st.warning("Attendance file not found!")

    elif action == "View Attendance":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            view_attendance()
        else:
            st.warning("Please log in first!")


def main():

    st.title("Face Recognition Attendance System (Nguyen Vu Huy)")

    option = st.sidebar.selectbox(
        "Select an option:", ["Face Recognition", "Admin Login"])

    if option == "Face Recognition":
        display_face_recognition()

    elif option == "Admin Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid credentials")

    if st.session_state.get('logged_in', False):
        admin_mode()
    else:
        st.write("Please log in as an admin to access admin functionalities.")


if __name__ == "__main__":
    main()
