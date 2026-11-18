# 📚 Smart Study Monitor — Complete Beginner's Guide

> A real-time webcam-based study assistant that keeps you focused by detecting sleep, face covering, and phone usage.

---

## 📖 Table of Contents

1. [What is this Project?](#1-what-is-this-project)
2. [How Does it Work? (Simple Explanation)](#2-how-does-it-work-simple-explanation)
3. [Technologies Used](#3-technologies-used)
4. [Project Files Explained](#4-project-files-explained)
5. [Understanding the Code (app.py)](#5-understanding-the-code-apppy)
6. [Step-by-Step Setup Guide](#6-step-by-step-setup-guide)
7. [How to Run the App](#7-how-to-run-the-app)
8. [What Happens When You Run It](#8-what-happens-when-you-run-it)
9. [Warnings and What They Mean](#9-warnings-and-what-they-mean)
10. [Troubleshooting Common Errors](#10-troubleshooting-common-errors)
11. [Glossary (Key Terms for Beginners)](#11-glossary-key-terms-for-beginners)

---

## 1. What is this Project?

**Smart Study Monitor** is a Python application that uses your **webcam** to watch over you while you study. It acts like a virtual study buddy that:

- 😴 **Detects if you fall asleep** — by tracking if your eyes stay closed for too long.
- 🙈 **Detects if you cover your face** — by checking if your face disappears from the camera.
- 📱 **Detects if you use your phone** — by identifying a mobile phone in the camera frame.

When any of these distractions are detected, the app:
- Shows a **text warning** on the screen.
- Plays a **loud alarm sound** to alert you.

### 🎯 Goal of the Project

To help students stay focused during study sessions using real-time computer vision technology — no human supervisor required!

---

## 2. How Does it Work? (Simple Explanation)

Think of this app like a security camera with a brain. Here's the simple flow:

```
Your Webcam
    ↓
[Capture a video frame]
    ↓
┌─────────────────────────────────────────┐
│  Check 1: Are your eyes closed too long? │  → 😴 Sleep Warning
│  Check 2: Is your face missing?          │  → 🙈 Face Cover Warning
│  Check 3: Is a phone visible?            │  → 📱 Phone Warning
└─────────────────────────────────────────┘
    ↓
Show warning on screen + Play alarm
    ↓
[Repeat for the next frame]
```

This process repeats **30+ times every second** (at your camera's frame rate), so detection is nearly real-time.

---

## 3. Technologies Used

| Library / Tool | What it Does | Why We Use It |
|---|---|---|
| **Python** | Main programming language | Easy to use, great for AI/ML projects |
| **OpenCV (`cv2`)** | Opens the webcam, reads frames, draws on screen | Industry standard for computer vision |
| **cvzone** | A helper library built on top of OpenCV | Makes face detection and drawing easier |
| **MediaPipe (via cvzone)** | Powers the face mesh detection | Google's fast and accurate face tracking |
| **Ultralytics YOLO** | Object detection model | Detects phones in the video frame |
| **pygame** | Plays audio files | Simple cross-platform audio playback |
| **YOLOv8n model (`yolov8n.pt`)** | Pre-trained AI model file | Already knows how to detect 80 types of objects |

### 🧠 What is AI/Machine Learning here?

Two AI models are used:
1. **Face Mesh (MediaPipe)** — Maps 468 points on your face to track eyes, lips, nose, etc.
2. **YOLOv8n (YOLO = You Only Look Once)** — A fast object detection model that can spot a "cell phone" among 80 object categories from the COCO dataset.

---

## 4. Project Files Explained

```
Smart_Study_Monitor-main/
│
├── app.py                  ← Main Python script (the brain of the app)
│
├── yolov8n.pt              ← Pre-trained YOLOv8 model for phone detection
│                             (Downloaded from Ultralytics, trained on COCO dataset)
│
├── face_landmarker.task    ← MediaPipe face landmark model file
│                             (Used internally by cvzone for face tracking)
│
├── hand_landmarker.task    ← MediaPipe hand landmark model file
│                             (Included but not actively used in this version)
│
├── alarm.mp3               ← Sound played when sleeping is detected 😴
├── faudio.mp3              ← Sound played when face is hidden 🙈
├── paudio.mp3              ← Sound played when phone is detected 📱
│
├── README.md               ← Quick start guide
└── DOCUMENTATION.md        ← This file (full beginner guide)
```

---

## 5. Understanding the Code (`app.py`)

Let's walk through the entire `app.py` file section by section so you understand every line.

### Section 1 — Imports (Loading the Tools)

```python
import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import pygame
```

**What this does:**
- `import cv2` → Loads OpenCV so we can use the webcam and draw on screen.
- `import cvzone` → Loads cvzone, which has helper functions for putting text on screen.
- `from cvzone.FaceMeshModule import FaceMeshDetector` → Loads the face mesh detector.
- `from ultralytics import YOLO` → Loads the YOLO object detection framework.
- `import pygame` → Loads pygame to play MP3 audio files.

---

### Section 2 — Initializing Audio

```python
pygame.mixer.init()
alarm_sleep    = pygame.mixer.Sound("alarm.mp3")   # Sleep alarm
alarm_facehide = pygame.mixer.Sound("faudio.mp3")  # Face hidden alarm
alarm_phone    = pygame.mixer.Sound("paudio.mp3")  # Phone alarm

current_playing = None  # Tracks which alarm is currently playing
```

**What this does:**
- `pygame.mixer.init()` → Starts the audio system.
- Three `Sound` objects are loaded from MP3 files. Each represents a different alarm.
- `current_playing` is a variable that tracks which alarm is active so we don't overlap sounds.

---

### Section 3 — Setting Up the Camera and Models

```python
cap = cv2.VideoCapture(0)
face_detector = FaceMeshDetector(maxFaces=1)
phone_detector = YOLO("yolov8n.pt")
classNames = phone_detector.names
```

**What this does:**
- `cv2.VideoCapture(0)` → Opens your default webcam (the `0` means "first webcam").
- `FaceMeshDetector(maxFaces=1)` → Creates a face detector that tracks one face at a time.
- `YOLO("yolov8n.pt")` → Loads the YOLO model from the file `yolov8n.pt`.
- `classNames` → A list of 80 object names the YOLO model can detect (like "cell phone", "car", "dog", etc.)

---

### Section 4 — Landmark Indices

```python
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145
FACE_LEFT       = 130
FACE_RIGHT      = 243
```

**What this does:**

The face mesh creates 468 numbered points (called "landmarks") on your face. These 4 numbers are specific points:

```
      [159] ← top of left eye
     /       \
[130]          [243]    ← left and right edges of face (width reference)
     \       /
      [145] ← bottom of left eye
```

By measuring the distance between point 159 and 145, we get the **eye opening height**.  
By measuring between 130 and 243, we get the **face width**.

---

### Section 5 — Thresholds (How Sensitive the Detection Is)

```python
closed_frames = 0
SLEEP_THRESHOLD_FRAMES = 15   # ~0.5 seconds of closed eyes = sleeping

covered_frames = 0
COVER_THRESHOLD_FRAMES = 20   # ~0.6 seconds of missing face = face covered
```

**What this does:**
- `closed_frames` → Counts how many consecutive frames had closed eyes.
- `SLEEP_THRESHOLD_FRAMES = 15` → If eyes are closed for 15 frames in a row → trigger sleep alert.
- `covered_frames` → Counts how many consecutive frames had no face detected.
- `COVER_THRESHOLD_FRAMES = 20` → If face is missing for 20 frames → trigger face cover alert.

> 💡 **Tip:** At 30 FPS (frames per second), 15 frames ≈ 0.5 seconds. So if your eyes close for half a second, it triggers the alarm.

---

### Section 6 — The Main Loop

```python
while True:
    success, img = cap.read()
    if not success:
        break
```

**What this does:**
- `while True` → Runs forever until the user presses `Q`.
- `cap.read()` → Reads one frame (a snapshot) from the webcam.
- `success` → A boolean (True/False) indicating if the frame was captured successfully.
- If the webcam fails, we `break` out of the loop and stop.

---

### Section 7 — Sleep & Face Cover Detection

```python
img, faces = face_detector.findFaceMesh(img, draw=False)

if faces:
    covered_frames = 0          # Face is visible → reset counter
    face = faces[0]             # Get the first face detected

    eye_dist, _  = face_detector.findDistance(face[159], face[145])
    face_dist, _ = face_detector.findDistance(face[130], face[243])

    ratio = (eye_dist / face_dist) * 100   # Eye Aspect Ratio (EAR)

    if ratio < 11.0:
        closed_frames += 1      # Eye is nearly closed
    else:
        closed_frames = 0       # Eye is open → reset

    if closed_frames >= SLEEP_THRESHOLD_FRAMES:
        is_sleepy = True        # Trigger sleep alarm!

else:
    closed_frames = 0
    covered_frames += 1         # Face missing → increment counter

    if covered_frames >= COVER_THRESHOLD_FRAMES:
        is_face_covered = True  # Trigger face cover alarm!
```

**What the Eye Aspect Ratio (EAR) means:**

```
        [159]
       /     \
   EAR = eye height / face width × 100
       \     /
        [145]

Open eye  → larger height → ratio ≈ 14–18
Closed eye → smaller height → ratio < 11
```

When the ratio drops **below 11.0**, the eye is considered closed. If this persists for **15 or more frames**, the person is considered asleep.

---

### Section 8 — Phone Detection

```python
results = phone_detector.predict(img, stream=True, verbose=False)

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])

        if classNames[cls_id] == "cell phone" and conf > 0.5:
            phone_detected = True

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cvzone.putTextRect(img, f"Phone detected! {int(conf*100)}%", ...)
```

**What this does:**
- The YOLO model scans the entire frame and finds all objects it can recognize.
- We look specifically for `"cell phone"` objects.
- `conf > 0.5` → Only count detections where the model is **more than 50% confident** it's a phone.
- A purple rectangle is drawn around the detected phone.
- Text shows the detection percentage (e.g., "Phone detected! 87%").

---

### Section 9 — Alarm Logic and Priorities

```python
if is_face_covered:                          # ← Highest priority
    show warning "DONT COVER YOUR FACE!"
    play alarm_facehide

elif is_sleepy:                              # ← Second priority
    show warning "WAKE UP & STUDY!"
    play alarm_sleep

elif phone_detected:                         # ← Third priority
    show warning "PUT THE PHONE AWAY!"
    play alarm_phone
```

**Priority order explained:**

| Priority | Condition | Why This Order? |
|---|---|---|
| 1 (Highest) | Face covered | A covered face means we can't even check for sleep |
| 2 | Sleeping | Sleep is more serious than phone use |
| 3 (Lowest) | Phone detected | Distraction but person is awake |

This `elif` chain ensures **only one alarm plays at a time**.

---

### Section 10 — Cleanup

```python
cap.release()
cv2.destroyAllWindows()
```

**What this does:**
- `cap.release()` → Releases the webcam so other apps can use it.
- `cv2.destroyAllWindows()` → Closes the camera window.

---

## 6. Step-by-Step Setup Guide

### Prerequisites Checklist

Before you start, make sure you have:
- [ ] A Windows 10 or 11 computer
- [ ] A working webcam (built-in or USB)
- [ ] Speakers or headphones (for the alarm sounds)
- [ ] Internet connection (for downloading Python and packages)

---

### Step 1 — Install Python 3.11

> ⚠️ **Important:** Use Python **3.11** specifically. Python 3.14 has compatibility issues with `pygame`.

**Option A — Download from website:**
Go to: https://www.python.org/downloads/release/python-3119/  
Download the Windows installer and run it.  
✅ Make sure to check **"Add Python to PATH"** during installation!

**Option B — Install via terminal:**
```powershell
winget install --id Python.Python.3.11 -e --source winget
```

Verify installation:
```powershell
py -3.11 --version
# Expected output: Python 3.11.x
```

---

### Step 2 — Open the Project Folder

Open PowerShell or Command Prompt and navigate to the project folder:
```powershell
cd "C:\path\to\Smart_Study_Monitor-main"
```

> Replace `C:\path\to\` with the actual location where you saved the project.

---

### Step 3 — Create a Virtual Environment

A virtual environment is an isolated Python space — it keeps project packages separate from your system Python.

```powershell
py -3.11 -m venv .venv
```

Then activate it:
```powershell
.\.venv\Scripts\activate
```

You'll see `(.venv)` appear at the start of your terminal prompt — that means it's active!

---

### Step 4 — Install Required Packages

With the virtual environment active, install all dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install opencv-python cvzone ultralytics pygame mediapipe
```

This will download and install:
- `opencv-python` — for webcam and image processing
- `cvzone` — for face mesh and text overlays
- `ultralytics` — for the YOLO phone detector
- `pygame` — for playing alarm sounds
- `mediapipe` — required by cvzone for face landmark detection

> ⏳ This may take a few minutes depending on your internet speed.

---

## 7. How to Run the App

Make sure you are inside the project folder with the virtual environment active:

```powershell
cd "C:\path\to\Smart_Study_Monitor-main"
.\.venv\Scripts\activate
python app.py
```

Or without virtual environment:
```powershell
py -3.11 app.py
```

A window titled **"Smart Study Monitor"** will open showing your webcam feed.

**To quit:** Press the **`Q`** key while the camera window is in focus.

---

## 8. What Happens When You Run It

Here's what you'll see on screen:

```
┌─────────────────────────────────────────┐
│                                         │
│   Eye Ratio: 14          [your face]    │
│                                         │
│   (If alert): WAKE UP & STUDY!          │
│                                         │
└─────────────────────────────────────────┘
```

- **"Eye Ratio: XX"** — Shows the current Eye Aspect Ratio value (14–18 = open, <11 = closed).
- If a **phone is detected**, a purple box appears around it with the confidence percentage.
- If an alert is triggered, a **colored warning text** appears near the top of the screen.

---

## 9. Warnings and What They Mean

| Warning Message | Trigger | Alarm File |
|---|---|---|
| `DONT COVER YOUR FACE!` | Face not visible for ~0.6 seconds | `faudio.mp3` |
| `WAKE UP & STUDY!` | Eyes closed for ~0.5 seconds | `alarm.mp3` |
| `PUT THE PHONE AWAY!` | Phone detected in frame with >50% confidence | `paudio.mp3` |

---

## 10. Troubleshooting Common Errors

### ❌ `ModuleNotFoundError: No module named 'cv2'`
```powershell
pip install opencv-python
```

### ❌ `ModuleNotFoundError: No module named 'cvzone'`
```powershell
pip install cvzone
```

### ❌ `ModuleNotFoundError: No module named 'mediapipe'`
```powershell
pip install mediapipe
```

### ❌ `ModuleNotFoundError: No module named 'pygame'`
```powershell
pip install pygame
```

### ❌ `pygame` errors / compatibility issues
Make sure you're running Python 3.11:
```powershell
py -3.11 app.py
```

### ❌ Webcam window doesn't open / Black screen
- Ensure your webcam is connected and not used by another app (Zoom, Teams, etc.)
- Go to Windows **Settings → Privacy → Camera** and allow apps to access the camera.
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `app.py` if you have multiple cameras.

### ❌ No alarm sound playing
- Check that your volume is turned up.
- Make sure `alarm.mp3`, `faudio.mp3`, and `paudio.mp3` exist in the same folder as `app.py`.

### ❌ `yolov8n.pt` not found
- The YOLO model file must be in the same folder as `app.py`.
- If missing, Ultralytics will attempt to download it automatically on first run (requires internet).

### ❌ Slow performance / lag
- Close other heavy applications while running.
- Ensure good lighting — poor lighting causes the face detector to work harder.
- The YOLO model runs on CPU by default; a GPU will make it significantly faster.

---

## 11. Glossary (Key Terms for Beginners)

| Term | Meaning |
|---|---|
| **Frame** | A single image captured from the webcam. Video is just many frames per second. |
| **FPS (Frames Per Second)** | How many images are captured every second. Typical webcams do 30 FPS. |
| **Computer Vision** | The field of AI that lets computers "see" and understand images/video. |
| **Face Mesh** | A grid of 468 points mapped onto a face to track its shape and movement. |
| **Landmark** | A specific numbered point on the face mesh (e.g., point 159 = top of left eye). |
| **Eye Aspect Ratio (EAR)** | A ratio that measures how open or closed an eye is. Lower = more closed. |
| **YOLO** | "You Only Look Once" — a fast AI model for detecting objects in images. |
| **COCO Dataset** | A large image dataset with 80 object categories used to train YOLO. |
| **Confidence Score** | A percentage (0–100%) showing how sure the AI is about a detection. |
| **Virtual Environment** | An isolated Python environment to keep project dependencies separate. |
| **pip** | Python's package manager — used to install libraries like `opencv-python`. |
| **MediaPipe** | A framework from Google for face/hand/pose tracking in real-time. |
| **pygame** | A Python library often used for games, but also great for playing audio. |
| **OpenCV** | Open Source Computer Vision library — the standard tool for image processing. |
| **`elif`** | Python shorthand for "else if" — used to check multiple conditions in order. |
| **Boolean** | A value that is either `True` or `False`. |
| **Threshold** | A cutoff value — if something goes above/below it, an action is triggered. |

---

## 🚀 Quick Start Summary

```powershell
# 1. Navigate to the project folder
cd "C:\path\to\Smart_Study_Monitor-main"

# 2. Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install opencv-python cvzone ultralytics pygame mediapipe

# 4. Run the app
python app.py

# Press Q to quit
```

---

*This documentation was written to help beginners understand the Smart Study Monitor project from the ground up. Happy studying! 🎓*
