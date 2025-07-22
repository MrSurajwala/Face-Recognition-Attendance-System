import tkinter as tk
from tkinter import messagebox, ttk
import customtkinter as ctk
import cv2
import face_recognition
import pickle
import os
import sqlite3
from datetime import datetime
import threading
import shutil
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
import configparser
import csv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import dlib
import numpy as np

# --- Constants and Global Setup ---
DATASET_PATH = "dataset"
ENCODINGS_FILE = "encodings.pickle"
DB_FILE = "attendance.db"
UNKNOWN_FACES_DIR = "unknown_faces"
CONFIG_FILE = "config.ini"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# --- UI Setup ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# --- Configuration Management ---
config = configparser.ConfigParser()

def load_config():
    if not os.path.exists(CONFIG_FILE):
        config['Settings'] = {'CaptureCount': '30'}
        config['Credentials'] = {'Username': 'admin', 'Password': 'admin'}
        with open(CONFIG_FILE, 'w') as configfile:
            config.write(configfile)
    config.read(CONFIG_FILE)
    return config

config = load_config()

# --- Helper Classes ---
class EnrollmentDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Enroll New Person"); self.geometry("350x200"); self.transient(parent); self.grab_set()
        self.name, self.student_id = "", ""
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Name:", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        self.name_entry = ctk.CTkEntry(self, placeholder_text="e.g., John Doe")
        self.name_entry.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="ew")
        ctk.CTkLabel(self, text="Student ID:", font=ctk.CTkFont(size=14)).grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.id_entry = ctk.CTkEntry(self, placeholder_text="e.g., 12345")
        self.id_entry.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        ok_button = ctk.CTkButton(self, text="OK", command=self.on_ok)
        ok_button.grid(row=2, column=1, padx=20, pady=(10,20), sticky="e")
        self.name_entry.focus()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
    def on_ok(self):
        self.name, self.student_id = self.name_entry.get().strip(), self.id_entry.get().strip()
        if not self.name or not self.student_id: messagebox.showwarning("Input Error", "Please enter both a name and a student ID.", parent=self); return
        self.destroy()
    def on_cancel(self): self.destroy()
    def get_input(self):
        self.wait_window()
        return self.name, self.student_id

class ChangePasswordDialog(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Change Password"); self.geometry("400x250"); self.transient(parent); self.grab_set()
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Current Password:").grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.current_pass_entry = ctk.CTkEntry(self, show="*")
        self.current_pass_entry.grid(row=0, column=1, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(self, text="New Password:").grid(row=1, column=0, padx=20, pady=10, sticky="w")
        self.new_pass_entry = ctk.CTkEntry(self, show="*")
        self.new_pass_entry.grid(row=1, column=1, padx=20, pady=10, sticky="ew")
        ctk.CTkLabel(self, text="Confirm New Password:").grid(row=2, column=0, padx=20, pady=10, sticky="w")
        self.confirm_pass_entry = ctk.CTkEntry(self, show="*")
        self.confirm_pass_entry.grid(row=2, column=1, padx=20, pady=10, sticky="ew")
        save_button = ctk.CTkButton(self, text="Save Password", command=self.save_password)
        save_button.grid(row=3, column=1, padx=20, pady=20, sticky="e")

    def save_password(self):
        current_pass, new_pass, confirm_pass = self.current_pass_entry.get(), self.new_pass_entry.get(), self.confirm_pass_entry.get()
        stored_password = config.get('Credentials', 'Password')
        if current_pass != stored_password: messagebox.showerror("Error", "Current password does not match.", parent=self); return
        if not new_pass or new_pass != confirm_pass: messagebox.showerror("Error", "New passwords do not match or are empty.", parent=self); return
        config.set('Credentials', 'Password', new_pass)
        with open(CONFIG_FILE, 'w') as configfile: config.write(configfile)
        messagebox.showinfo("Success", "Password changed successfully.", parent=self); self.destroy()

# --- Main Application ---
class FaceRecognitionApp(ctk.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Face Recognition Attendance System"); self.geometry("700x500"); self.resizable(False, False)

        if not os.path.exists(SHAPE_PREDICTOR_PATH):
            messagebox.showerror("Error", f"'{SHAPE_PREDICTOR_PATH}' not found. Please download it and place it in the project folder.")
            self.after(100, self.destroy); return

        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.grid_columnconfigure(0, weight=1); self.grid_rowconfigure(1, weight=1)
        self.image_references = []
        self.load_icons()

        title_label = ctk.CTkLabel(self, text="Face Recognition Attendance System", font=ctk.CTkFont(size=24, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.tab_view = ctk.CTkTabview(self, anchor="w"); self.tab_view.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.tab_view.add("Dashboard"); self.tab_view.add("Management"); self.tab_view.add("Settings")

        self.create_dashboard_tab(); self.create_management_tab(); self.create_settings_tab()

        self.status_label = ctk.CTkLabel(self, text="System Ready", font=ctk.CTkFont(size=12, slant="italic"))
        self.status_label.grid(row=2, column=0, padx=20, pady=(0, 10), sticky="w")
        self.update_dashboard_stats()

    def load_icons(self):
        try:
            self.icon_start = ctk.CTkImage(Image.open("start_icon.png"), size=(20, 20))
            self.icon_enroll = ctk.CTkImage(Image.open("enroll_icon.png"), size=(20, 20))
            self.icon_train = ctk.CTkImage(Image.open("train_icon.png"), size=(20, 20))
            self.icon_enroll_list = ctk.CTkImage(Image.open("enroll_list_icon.png"), size=(20, 20))
            self.icon_log = ctk.CTkImage(Image.open("log_icon.png"), size=(20, 20))
            self.icon_settings = ctk.CTkImage(Image.open("settings_icon.png"), size=(20, 20))
        except FileNotFoundError as e:
            messagebox.showwarning("Warning", f"Icon file not found: {e.filename}. Buttons will appear without icons.")
            self.icon_start = self.icon_enroll = self.icon_train = self.icon_enroll_list = self.icon_log = self.icon_settings = None

    def create_dashboard_tab(self):
        dashboard_tab = self.tab_view.tab("Dashboard"); dashboard_tab.grid_columnconfigure(0, weight=1)
        stats_frame = ctk.CTkFrame(dashboard_tab); stats_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        stats_frame.grid_columnconfigure((0, 1), weight=1)
        self.enrolled_label = ctk.CTkLabel(stats_frame, text="Total Enrolled: 0", font=ctk.CTkFont(size=16))
        self.enrolled_label.grid(row=0, column=0, padx=10, pady=10)
        self.today_label = ctk.CTkLabel(stats_frame, text="Attendance Today: 0", font=ctk.CTkFont(size=16))
        self.today_label.grid(row=0, column=1, padx=10, pady=10)
        action_frame = ctk.CTkFrame(dashboard_tab, fg_color="transparent")
        action_frame.grid(row=1, column=0, padx=20, pady=40)
        self.start_btn = ctk.CTkButton(action_frame, text="Start Live Attendance", height=60, font=ctk.CTkFont(size=18, weight="bold"),
                                       image=self.icon_start, compound="left", command=self.start_attendance)
        self.start_btn.pack()

    def create_management_tab(self):
        management_tab = self.tab_view.tab("Management"); management_tab.grid_columnconfigure((0, 1), weight=1)
        self.enroll_btn = ctk.CTkButton(management_tab, text="Enroll New Person", height=50, image=self.icon_enroll, compound="left", command=self.enroll_person)
        self.enroll_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.train_btn = ctk.CTkButton(management_tab, text="Train System", height=50, image=self.icon_train, compound="left", command=self.train_system)
        self.train_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.view_enroll_btn = ctk.CTkButton(management_tab, text="View Enrollments", height=50, image=self.icon_enroll_list, compound="left", command=self.view_enrollments)
        self.view_enroll_btn.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.view_log_btn = ctk.CTkButton(management_tab, text="View Attendance Log", height=50, image=self.icon_log, compound="left", command=self.view_attendance_log)
        self.view_log_btn.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(management_tab, mode='determinate')
        self.progress_bar.grid(row=2, column=0, columnspan=2, padx=10, pady=20, sticky="ew")
        self.progress_bar.set(0)

    def create_settings_tab(self):
        settings_tab = self.tab_view.tab("Settings"); settings_tab.grid_columnconfigure(1, weight=1)
        settings_frame = ctk.CTkFrame(settings_tab); settings_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        settings_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(settings_frame, text="Images to Capture:", font=ctk.CTkFont(size=14)).grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.capture_count_entry = ctk.CTkEntry(settings_frame)
        self.capture_count_entry.insert(0, config.get('Settings', 'CaptureCount', fallback='30'))
        self.capture_count_entry.grid(row=0, column=1, padx=20, pady=10, sticky="ew")
        save_settings_button = ctk.CTkButton(settings_frame, text="Save Settings", command=self.save_app_settings)
        save_settings_button.grid(row=1, column=1, padx=20, pady=10, sticky="e")
        password_frame = ctk.CTkFrame(settings_tab); password_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        password_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(password_frame, text="Security", font=ctk.CTkFont(size=16, weight="bold")).grid(row=0, column=0, columnspan=2, padx=20, pady=10)
        change_pass_button = ctk.CTkButton(password_frame, text="Change Admin Password", command=self.change_password)
        change_pass_button.grid(row=1, column=1, padx=20, pady=10, sticky="e")
        
    def save_app_settings(self):
        capture_count = self.capture_count_entry.get()
        if capture_count.isdigit() and int(capture_count) > 0:
            config.set('Settings', 'CaptureCount', capture_count)
            with open(CONFIG_FILE, 'w') as configfile: config.write(configfile)
            messagebox.showinfo("Success", "Settings saved successfully.")
        else: messagebox.showerror("Error", "Please enter a valid number for capture count.")

    def change_password(self): ChangePasswordDialog(self)

    def update_dashboard_stats(self):
        try: enrolled_count = len([name for name in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, name))])
        except Exception: enrolled_count = 0
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(DISTINCT student_id) FROM attendance WHERE date = ?", (today,))
                today_count = cursor.fetchone()[0]
        except Exception: today_count = 0
        self.enrolled_label.configure(text=f"Total Enrolled: {enrolled_count}")
        self.today_label.configure(text=f"Attendance Today: {today_count}")

    def update_status(self, message): self.status_label.configure(text=message)

    def set_buttons_state(self, state):
        for btn in [self.enroll_btn, self.train_btn, self.start_btn, self.view_enroll_btn, self.view_log_btn]:
            if btn: btn.configure(state=state)

    def enroll_person(self):
        dialog = EnrollmentDialog(self)
        name, student_id = dialog.get_input()
        if not name or not student_id: self.update_status("Enrollment cancelled."); return
        person_dir = os.path.join(DATASET_PATH, student_id)
        if os.path.exists(person_dir): messagebox.showerror("Enrollment Error", f"Student ID '{student_id}' already exists.", parent=self); return
        os.makedirs(person_dir)
        capture_count = config.getint('Settings', 'CaptureCount', fallback=30)
        messagebox.showinfo("Camera Starting", f"Look at the camera. Capturing {capture_count} images for {name}.", parent=self)
        self.set_buttons_state("disabled")
        threading.Thread(target=self._capture_thread, args=(name, student_id, capture_count), daemon=True).start()

    def _capture_thread(self, name, student_id, capture_count):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): self.after(0, lambda: messagebox.showerror("Error", "Could not open webcam.", parent=self)); self.after(0, self.set_buttons_state, "normal"); return
        person_dir = os.path.join(DATASET_PATH, student_id)
        count = 0
        while count < capture_count:
            ret, frame = cap.read()
            if not ret: break
            cv2.putText(frame, f"Capturing: {count+1}/{capture_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Enrollment", frame)
            filename = f"{name.replace(' ', '_')}_{student_id}_{count}.jpg"
            cv2.imwrite(os.path.join(person_dir, filename), frame)
            count += 1
            self.after(0, self.update_status, f"Captured {count}/{capture_count} images")
            if cv2.waitKey(100) & 0xFF == ord('q'): break
        cap.release(); cv2.destroyAllWindows()
        self.after(0, self.update_status, f"Enrollment for {name} complete.")
        self.after(0, self.set_buttons_state, "normal")
        self.after(0, self.update_dashboard_stats)
        self.after(0, lambda: messagebox.showinfo("Success", "Enrollment complete! Please re-train the system.", parent=self))

    def train_system(self):
        self.set_buttons_state("disabled"); self.progress_bar.set(0)
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            imagePaths = list(paths.list_images(DATASET_PATH))
            if not imagePaths:
                if os.path.exists(ENCODINGS_FILE): os.remove(ENCODINGS_FILE)
                self.after(0, lambda: messagebox.showinfo("Info", "Dataset is empty. Trained data has been cleared.", parent=self)); self.after(0, self.set_buttons_state, "normal"); return
            knownData = {}
            for i, imagePath in enumerate(imagePaths):
                self.after(0, self.update_status, f"Processing image {i + 1}/{len(imagePaths)}...")
                self.after(0, self.progress_bar.set, (i + 1) / len(imagePaths))
                student_id = os.path.basename(os.path.dirname(imagePath))
                name = os.path.basename(imagePath).split('_')[0].replace('_', ' ')
                if student_id not in knownData: knownData[student_id] = {'name': name, 'encodings': []}
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb, model="hog")
                encodings = face_recognition.face_encodings(rgb, boxes)
                knownData[student_id]['encodings'].extend(encodings)
            with open(ENCODINGS_FILE, "wb") as f: pickle.dump(knownData, f)
            self.after(0, self.update_status, "Training complete.")
            self.after(0, lambda: messagebox.showinfo("Success", "System training is complete.", parent=self))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"An error occurred during training: {e}", parent=self))
        finally:
            self.after(0, self.progress_bar.set, 0)
            self.after(0, self.set_buttons_state, "normal")

    def start_attendance(self):
        try:
            with open(ENCODINGS_FILE, "rb") as f: self.knownData = pickle.load(f)
            if not self.knownData: raise FileNotFoundError
        except (FileNotFoundError, EOFError): messagebox.showerror("Error", "No students have been trained.", parent=self); return
        self.all_encodings = [enc for data in self.knownData.values() for enc in data['encodings']]
        self.all_ids = [sid for sid, data in self.knownData.items() for _ in data['encodings']]
        if not self.all_encodings: messagebox.showerror("Error", "Training data is invalid or empty. Please re-train.", parent=self); return
        self.set_buttons_state("disabled")
        threading.Thread(target=self._attendance_thread, daemon=True).start()

    def _attendance_thread(self):
        EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES = 0.25, 3
        BLINK_COUNTER, TOTAL_BLINKS = 0, 0
        liveness_confirmed = False
        (lStart, lEnd), (rStart, rEnd) = (42, 48), (36, 42)
        cap = cv2.VideoCapture(0)
        self.after(0, self.update_status, "Live attendance started...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY, student_id TEXT, name TEXT, timestamp TEXT, date TEXT)''')
        conn.commit()
        process_this_frame = True
        def mark_attendance(student_id):
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute("SELECT * FROM attendance WHERE student_id = ? AND date = ?", (student_id, today))
            if cursor.fetchone() is None:
                name = self.knownData[student_id]['name']
                timestamp = datetime.now().strftime("%H:%M:%S")
                cursor.execute("INSERT INTO attendance (student_id, name, timestamp, date) VALUES (?, ?, ?, ?)", (student_id, name, timestamp, today))
                conn.commit()
                return f"Attendance marked for {name}."
            else: return f"{self.knownData[student_id]['name']} already marked."
        while True:
            ret, frame = cap.read()
            if not ret: break
            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                found_known_face = False
                if not liveness_confirmed:
                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                    boxes = face_recognition.face_locations(rgb_small_frame, model="hog")
                    rects = dlib.rectangles()
                    for (top, right, bottom, left) in boxes: rects.append(dlib.rectangle(left=left, top=top, right=right, bottom=bottom))
                    if rects:
                        shape = self.predictor(gray, rects[0])
                        shape_np = np.zeros((shape.num_parts, 2), dtype="int")
                        for i in range(0, shape.num_parts): shape_np[i] = (shape.part(i).x, shape.part(i).y)
                        leftEye, rightEye = shape_np[lStart:lEnd], shape_np[rStart:rEnd]
                        def eye_aspect_ratio(eye):
                            A = dist.euclidean(eye[1], eye[5]); B = dist.euclidean(eye[2], eye[4]); C = dist.euclidean(eye[0], eye[3])
                            return (A + B) / (2.0 * C)
                        leftEAR, rightEAR = eye_aspect_ratio(leftEye), eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        leftEyeHull, rightEyeHull = cv2.convexHull(leftEye), cv2.convexHull(rightEye)
                        cv2.drawContours(frame, [(leftEyeHull * 4)], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [(rightEyeHull * 4)], -1, (0, 255, 0), 1)
                        if ear < EYE_AR_THRESH: BLINK_COUNTER += 1
                        else:
                            if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES: TOTAL_BLINKS += 1
                            BLINK_COUNTER = 0
                        if TOTAL_BLINKS >= 2: liveness_confirmed = True; self.after(0, self.update_status, "Liveness confirmed. Now recognizing.")
                elif liveness_confirmed:
                    boxes = face_recognition.face_locations(rgb_small_frame, model="hog")
                    encodings = face_recognition.face_encodings(rgb_small_frame, boxes)
                    for (top, right, bottom, left), encoding in zip(boxes, encodings):
                        matches = face_recognition.compare_faces(self.all_encodings, encoding, tolerance=0.5)
                        name, student_id = "Unknown", "N/A"
                        if True in matches:
                            student_id = self.all_ids[matches.index(True)]
                            name = self.knownData[student_id]['name']
                            status_message = mark_attendance(student_id)
                            self.after(0, self.update_status, status_message)
                            found_known_face = True
                        if name == "Unknown": cv2.imwrite(os.path.join(UNKNOWN_FACES_DIR, f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"), frame)
                        top *= 4; right *= 4; bottom *= 4; left *= 4
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, f"{name} ({student_id})", (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            process_this_frame = not process_this_frame
            if not liveness_confirmed:
                cv2.putText(frame, f"Blinks: {TOTAL_BLINKS}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"EAR: {ear:.2f}" if 'ear' in locals() else "EAR: N/A", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if found_known_face: cv2.putText(frame, "Success!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Live Attendance", frame)
            if (found_known_face and liveness_confirmed) or (cv2.waitKey(1) & 0xFF == ord('q')):
                if found_known_face: cv2.waitKey(2000)
                break
        cap.release(); cv2.destroyAllWindows(); conn.close()
        self.after(0, self.update_status, "System Ready")
        self.after(0, self.set_buttons_state, "normal")
        self.after(0, self.update_dashboard_stats)

    # ###############################################################
    # ##               MODIFIED FUNCTION STARTS HERE               ##
    # ###############################################################
    def view_enrollments(self):
        enroll_window = ctk.CTkToplevel(self)
        enroll_window.title("Enrolled Students"); enroll_window.geometry("650x500"); enroll_window.transient(self); enroll_window.grab_set()
        style = ttk.Style(); style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", rowheight=50)
        style.configure("Treeview.Heading", background="#565b5e", foreground="white", font=('Calibri', 10, 'bold'))
        style.map('Treeview', background=[('selected', '#347083')])
        tree_frame = ctk.CTkFrame(enroll_window); tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        tree = ttk.Treeview(tree_frame, columns=("ID", "Name"), show="headings tree")
        tree.heading("#0", text="Photo"); tree.column("#0", width=70, anchor=tk.CENTER, stretch=False)
        tree.heading("ID", text="Student ID"); tree.column("ID", width=200, anchor=tk.CENTER)
        tree.heading("Name", text="Name"); tree.column("Name", width=300)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar = ctk.CTkScrollbar(tree_frame, command=tree.yview); scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        
        self.image_references.clear()
        for student_id in sorted(os.listdir(DATASET_PATH)):
            student_dir = os.path.join(DATASET_PATH, student_id)
            if os.path.isdir(student_dir):
                name = "N/A"
                photo = None
                try:
                    image_files = [f for f in os.listdir(student_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if image_files:
                        first_image_path = os.path.join(student_dir, image_files[0])
                        name = os.path.basename(first_image_path).split('_')[0].replace('_', ' ')
                        img = Image.open(first_image_path); img.thumbnail((40, 40)); photo = ImageTk.PhotoImage(img)
                        self.image_references.append(photo)
                    tree.insert("", tk.END, image=photo, values=(student_id, name))
                except (IndexError, FileNotFoundError):
                    tree.insert("", tk.END, text="No Img", values=(student_id, name))
                    continue
        
        button_frame = ctk.CTkFrame(enroll_window); button_frame.pack(fill="x", padx=10, pady=(0, 10))
        def view_student_log():
            if not tree.selection(): messagebox.showwarning("Warning", "Please select a student.", parent=enroll_window); return
            student_id = tree.item(tree.selection()[0], 'values')[0]
            self.view_attendance_log(student_id=student_id)
        def delete_enrollment():
            if not tree.selection(): return
            if messagebox.askyesno("Confirm", "This will delete all images for the student. Continue?", parent=enroll_window):
                for item in tree.selection():
                    student_id = tree.item(item, 'values')[0]
                    try:
                        shutil.rmtree(os.path.join(DATASET_PATH, student_id)); tree.delete(item)
                        self.update_status(f"Deleted enrollment for ID {student_id}.")
                        self.update_dashboard_stats()
                        messagebox.showinfo("Important", "Please re-train the system.", parent=enroll_window)
                    except Exception as e: messagebox.showerror("Error", f"Could not delete: {e}", parent=enroll_window)
        view_log_button = ctk.CTkButton(button_frame, text="View Attendance", command=view_student_log)
        view_log_button.pack(side="left", padx=10, pady=5)
        delete_button = ctk.CTkButton(button_frame, text="Delete Enrollment", command=delete_enrollment, fg_color="#c0392b", hover_color="#a93226")
        delete_button.pack(side="right", padx=10, pady=5)
    # ###############################################################
    # ##                MODIFIED FUNCTION ENDS HERE                ##
    # ###############################################################

    def view_attendance_log(self, student_id=None):
        log_window = ctk.CTkToplevel(self)
        log_window.title("Attendance Log"); log_window.geometry("950x500"); log_window.grab_set()
        filter_frame = ctk.CTkFrame(log_window); filter_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(filter_frame, text="Start Date (YYYY-MM-DD):").pack(side="left", padx=(10,5))
        start_date_entry = ctk.CTkEntry(filter_frame); start_date_entry.pack(side="left")
        ctk.CTkLabel(filter_frame, text="End Date (YYYY-MM-DD):").pack(side="left", padx=(10,5))
        end_date_entry = ctk.CTkEntry(filter_frame); end_date_entry.pack(side="left")
        tree_frame = ctk.CTkFrame(log_window); tree_frame.pack(fill="both", expand=True, padx=10, pady=5)
        style = ttk.Style(); style.theme_use("default")
        style.configure("Treeview", background="#2b2b2b", foreground="white", fieldbackground="#2b2b2b", rowheight=25)
        style.configure("Treeview.Heading", background="#565b5e", foreground="white", font=('Calibri', 10, 'bold'))
        style.map('Treeview', background=[('selected', '#347083')])
        tree = ttk.Treeview(tree_frame, columns=("ID", "Student ID", "Name", "Date", "Time"), show="headings")
        tree.heading("ID", text="Record ID"); tree.column("ID", width=70, anchor=tk.CENTER)
        tree.heading("Student ID", text="Student ID"); tree.column("Student ID", width=120, anchor=tk.CENTER)
        tree.heading("Name", text="Name"); tree.column("Name", width=200)
        tree.heading("Date", text="Date"); tree.column("Date", width=120, anchor=tk.CENTER)
        tree.heading("Time", text="Time"); tree.column("Time", width=120, anchor=tk.CENTER)
        tree.pack(side="left", fill="both", expand=True)
        scrollbar = ctk.CTkScrollbar(tree_frame, command=tree.yview); scrollbar.pack(side="right", fill="y")
        tree.configure(yscrollcommand=scrollbar.set)
        def populate_tree(query, params=()):
            for item in tree.get_children(): tree.delete(item)
            try:
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    for record in cursor.fetchall(): tree.insert("", tk.END, values=record)
            except Exception as e: messagebox.showerror("Error", f"Could not load data: {e}", parent=log_window)
        def apply_filter():
            start, end = start_date_entry.get(), end_date_entry.get()
            base_query = "SELECT id, student_id, name, date, timestamp FROM attendance"
            conditions, params = [], []
            if student_id: conditions.append("student_id = ?"), params.append(student_id)
            if start and end: conditions.append("date BETWEEN ? AND ?"), params.extend([start, end])
            if conditions: base_query += " WHERE " + " AND ".join(conditions)
            base_query += " ORDER BY id DESC"
            populate_tree(base_query, tuple(params))
        initial_query, initial_params = "SELECT id, student_id, name, date, timestamp FROM attendance ORDER BY id DESC", ()
        if student_id:
            initial_query, initial_params = "SELECT id, student_id, name, date, timestamp FROM attendance WHERE student_id = ? ORDER BY id DESC", (student_id,)
        populate_tree(initial_query, initial_params)
        filter_button = ctk.CTkButton(filter_frame, text="Filter", command=apply_filter); filter_button.pack(side="left", padx=10)
        reset_button = ctk.CTkButton(filter_frame, text="Reset", command=lambda: populate_tree("SELECT id, student_id, name, date, timestamp FROM attendance ORDER BY id DESC")); reset_button.pack(side="left")
        button_frame = ctk.CTkFrame(log_window); button_frame.pack(fill="x", padx=10, pady=5)
        def delete_record():
            if not tree.selection(): return
            if messagebox.askyesno("Confirm", "Delete selected record(s)?", parent=log_window):
                for item in tree.selection():
                    record_id = tree.item(item, 'values')[0]
                    try:
                        with sqlite3.connect(DB_FILE) as conn: conn.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
                        tree.delete(item)
                    except Exception as e: messagebox.showerror("Error", f"Failed to delete: {e}", parent=log_window)
        def export_to_csv():
            if not tree.get_children(): return
            filepath = tk.filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if not filepath: return
            try:
                with open(filepath, "w", newline="", encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([tree.heading(c)["text"] for c in tree["columns"]])
                    for item in tree.get_children(): writer.writerow(tree.item(item)["values"])
                messagebox.showinfo("Success", "Data exported successfully.", parent=log_window)
            except Exception as e: messagebox.showerror("Error", f"Failed to export: {e}", parent=log_window)
        def generate_pdf():
            if not tree.get_children(): return
            filepath = tk.filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if not filepath: return
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            data = [[tree.heading(c)["text"] for c in tree["columns"]]] + [tree.item(item)["values"] for item in tree.get_children()]
            table = Table(data)
            table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey),('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                                       ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                                       ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                                       ('GRID', (0,0), (-1,-1), 1, colors.black)]))
            doc.build([table])
            messagebox.showinfo("Success", "PDF report generated.", parent=log_window)
        delete_button = ctk.CTkButton(button_frame, text="Delete Selected", command=delete_record, fg_color="#c0392b", hover_color="#a93226"); delete_button.pack(side="left", padx=5)
        export_button = ctk.CTkButton(button_frame, text="Export to CSV", command=export_to_csv); export_button.pack(side="left", padx=5)
        pdf_button = ctk.CTkButton(button_frame, text="Generate PDF Report", command=generate_pdf); pdf_button.pack(side="left", padx=5)

class LoginWindow(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Admin Login"); self.geometry("300x200"); self.resizable(False, False)
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="Username").pack(pady=(20,0))
        self.user_entry = ctk.CTkEntry(self, placeholder_text="admin")
        self.user_entry.pack()
        ctk.CTkLabel(self, text="Password").pack(pady=(10,0))
        self.pass_entry = ctk.CTkEntry(self, placeholder_text="admin", show="*")
        self.pass_entry.pack()
        ctk.CTkButton(self, text="Login", command=self.login).pack(pady=20)
    
    def login(self):
        config.read(CONFIG_FILE)
        username = config.get('Credentials', 'Username')
        password = config.get('Credentials', 'Password')
        if self.user_entry.get() == username and self.pass_entry.get() == password:
            self.destroy()
            app = FaceRecognitionApp()
            app.mainloop()
        else:
            messagebox.showerror("Login Failed", "Invalid username or password.")

if __name__ == "__main__":
    class SimplePaths:
        def list_images(self, base_path):
            for root, _, filenames in os.walk(base_path):
                for filename in filenames:
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        yield os.path.join(root, filename)
    paths = SimplePaths()
    login_app = LoginWindow()
    login_app.mainloop()