import cv2
import numpy as np
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog, messagebox
import os


def init_db():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance 
                 (name TEXT, date TEXT, time TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS known_faces 
                 (name TEXT, id INTEGER PRIMARY KEY AUTOINCREMENT)''')
    conn.commit()
    conn.close()


recognizer = cv2.face.LBPHFaceRecognizer_create()


def train_recognizer():
    known_face_ids = []
    known_face_names = []
    faces = []
    labels = []

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT id, name FROM known_faces")
    known_faces = c.fetchall()

    for (face_id, name) in known_faces:
        img_path = f"faces/{face_id}.jpg"
        if os.path.exists(img_path):
            face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            faces.append(face_img)
            labels.append(face_id)
            known_face_names.append(name)
            known_face_ids.append(face_id)

    if faces and labels:
        recognizer.train(faces, np.array(labels))

    conn.close()
    return known_face_names, known_face_ids


def save_new_face(name, face_img):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("INSERT INTO known_faces (name) VALUES (?)", (name,))
    face_id = c.lastrowid
    conn.commit()
    conn.close()


    os.makedirs("faces", exist_ok=True)
    cv2.imwrite(f"faces/{face_id}.jpg", face_img)

    train_recognizer()

# Mark attendance
def mark_attendance(name):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    c.execute("INSERT INTO attendance (name, date, time) VALUES (?, ?, ?)", (name, date, time))
    conn.commit()
    conn.close()


def add_new_face():

    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))


        for (x, y, w, h) in faces:
            face_img = gray_frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (200, 200))


            root = tk.Tk()
            root.withdraw()
            name = simpledialog.askstring("Add New Face", "Enter the name for the new face:")
            root.destroy()

            if name:
                save_new_face(name, face_img_resized)
                messagebox.showinfo("Face Added", f"{name} has been added successfully.")
            video_capture.release()
            cv2.destroyAllWindows()
            return

        cv2.imshow('Add New Face - Press Q to Cancel', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def recognize_faces():
    known_face_names, known_face_ids = train_recognizer()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = gray_frame[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (200, 200))


            label, confidence = recognizer.predict(face_img_resized)
            if confidence < 80:
                name = known_face_names[known_face_ids.index(label)]
                mark_attendance(name)
            else:
                name = "Unknown"


            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Attendance System - Press Q to Exit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def open_attendance_system():
    recognize_faces()

def main():
    init_db()
    root = tk.Tk()
    root.title("Face Recognition System")
    root.geometry("400x300")

    tk.Label(root, text="Face Recognition", font=("Helvetica", 16)).pack(pady=20)

    start_button = tk.Button(root, text="Start Recognition", command=open_attendance_system, font=("Helvetica", 14))
    start_button.pack(pady=20)

    add_face_button = tk.Button(root, text="Add New Face", command=add_new_face, font=("Helvetica", 14))
    add_face_button.pack(pady=20)

    quit_button = tk.Button(root, text="exit", command=root.quit, font=("Helvetica", 14))
    quit_button.pack(pady=20)

    root.mainloop()

main()
