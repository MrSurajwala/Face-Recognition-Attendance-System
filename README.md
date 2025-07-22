# Face Recognition Attendance System

## Description

This project is a complete, standalone Face Recognition Attendance System built with Python. It is designed to replace manual, error-prone attendance methods in schools and offices with a modern, automated solution that is fast, secure, and efficient.


## Key Features

-   🔐 **Secure Admin Login:** Protects all student data and administrative functions with a configurable password.
-   🧑‍🎓 **Robust Enrollment:** A guided process for enrolling new students with a unique ID, preventing duplicate entries.
-   🧠 **One-Click Training:** The system learns new faces with the click of a button and clears its memory if no students are enrolled.
-   👁️ **Liveness Detection:** An essential anti-spoofing feature that requires users to blink twice before attendance is marked, preventing cheating with static photos.
-   🚀 **Fast & Optimized Attendance:** Uses frame skipping and image resizing for a smooth, responsive experience during real-time recognition.
-   📊 **Dashboard with Live Stats:** The home screen displays key metrics, including "Total Students Enrolled" and "Attendance Marked Today."
-   📜 **Advanced Reporting & Management:**
    -   View all enrolled students with their profile photos.
    -   Delete a student's entire record.
    -   View a complete attendance log with filters for date ranges and specific students.
    -   Export attendance data to **CSV (Excel)** and printable **PDF** formats.

## Setup Instructions

To set up and run this project on a new computer, follow these steps:

1.  **Prerequisites:**
    * Ensure you have Python (version 3.8 or newer) installed.
    * Have the necessary C++ build tools for `dlib` installation.

2.  **Install Libraries:**
    Open your terminal or command prompt and install the required libraries:
    ```bash
    pip install opencv-python face-recognition dlib customtkinter Pillow scipy reportlab numpy
    ```

3.  **Download Shape Predictor Model:**
    * Download the pre-trained model from [this link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
    * Unzip the file to get `shape_predictor_68_face_landmarks.dat`.
    * Place this `.dat` file in the same folder as the `gui.py` script.

4.  **Run the Application:**
    ```bash
    python gui.py
    ```
    On the first run, the system will create a `config.ini` file. The default login is `admin` / `admin`.

## Technology Stack

-   **Core:** Python, OpenCV, face-recognition, dlib, NumPy
-   **UI:** CustomTkinter, Pillow (PIL)
-   **Data:** SQLite, Pickle, ConfigParser
-   **Reporting:** CSV, ReportLab
