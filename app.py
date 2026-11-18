import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import pygame

# Initialize Pygame Mixer with pre_init to ensure correct sample rate for MP3s
pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()

# Verify mixer initialized correctly
if not pygame.mixer.get_init():
    print("[WARNING] pygame mixer failed to initialize! Trying fallback settings...")
    pygame.mixer.quit()
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

print(f"[INFO] Mixer initialized: {pygame.mixer.get_init()}")

# Allocate dedicated channels to prevent conflicts between alarms
pygame.mixer.set_num_channels(16)
CH_SLEEP    = pygame.mixer.Channel(0)
CH_FACEHIDE = pygame.mixer.Channel(1)
CH_PHONE    = pygame.mixer.Channel(2)

# Load and configure alarm sounds
alarm_sleep    = pygame.mixer.Sound("alarm.mp3")    # Sleep alarm
alarm_facehide = pygame.mixer.Sound("faudio.mp3")   # Face hidden/covered alarm
alarm_phone    = pygame.mixer.Sound("paudio.mp3")   # Phone alarm

# Ensure full volume on all alarms
alarm_sleep.set_volume(1.0)
alarm_facehide.set_volume(1.0)
alarm_phone.set_volume(1.0)

# Track active audio state ('sleep', 'facehide', 'phone', or None)
current_playing = None

# Initialize Camera, Face Mesh, and YOLO Model
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[WARNING] Camera index 0 not available. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] No camera found. Please connect a camera and restart.")
        exit()

# Configure camera resolution and frame rate for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print(f"[INFO] Camera opened successfully — Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(cap.get(cv2.CAP_PROP_FPS))}fps")
face_detector = FaceMeshDetector(maxFaces=1)
phone_detector = YOLO("yolov8n.pt")  # COCO pre-trained lightweight model

# COCO Dataset Class Names
classNames = phone_detector.names  # Class '67' is 'cell phone' in standard COCO

# Landmark Indices for Sleep Detection
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
FACE_LEFT = 130
FACE_RIGHT = 243

# Frame Counters & Thresholds
closed_frames = 0
SLEEP_THRESHOLD_FRAMES = 15

covered_frames = 0
COVER_THRESHOLD_FRAMES = 20  # ~0.6 seconds of missing face triggers warning

while True:
    success, img = cap.read()
    if not success:
        print("[ERROR] Failed to read frame from camera. Check if the camera is connected.")
        break

    # Check active channel states
    is_sleep_busy    = CH_SLEEP.get_busy()
    is_facehide_busy = CH_FACEHIDE.get_busy()
    is_phone_busy    = CH_PHONE.get_busy()
    is_audio_busy    = is_sleep_busy or is_facehide_busy or is_phone_busy

    # Update current_playing based on channel activity
    if is_facehide_busy:
        current_playing = 'facehide'
    elif is_sleep_busy:
        current_playing = 'sleep'
    elif is_phone_busy:
        current_playing = 'phone'
    else:
        current_playing = None

    # ------------------- 1. SLEEP & FACE COVER DETECTION -------------------
    img, faces = face_detector.findFaceMesh(img, draw=False)
    is_sleepy = False
    is_face_covered = False

    if faces:
        # Face visible -> Reset covered frames counter
        covered_frames = 0
        face = faces[0]

        eye_dist, _ = face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
        face_dist, _ = face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])

        # Eye Aspect Ratio calculation
        ratio = (eye_dist / face_dist) * 100

        if ratio < 11.0:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames >= SLEEP_THRESHOLD_FRAMES:
            is_sleepy = True

        cvzone.putTextRect(img, f"Eye Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1)

    else:
        # Face missing/covered -> Increment covered frames counter
        closed_frames = 0
        covered_frames += 1
        
        if covered_frames >= COVER_THRESHOLD_FRAMES:
            is_face_covered = True

    # ------------------- 2. PHONE DETECTION -------------------
    results = phone_detector.predict(img, stream=True, verbose=False)
    phone_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if classNames[cls_id] == "cell phone" and conf > 0.5:
                phone_detected = True
                
                # Bounding box visual
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cvzone.putTextRect(img, f"Phone detected! {int(conf*100)}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(255, 0, 255))

    # ------------------- 3. ALARM LOGIC & DISPLAY -------------------

    # Priority 1: Face covered warning
    if is_face_covered:
        cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        if not CH_FACEHIDE.get_busy():      # Only play if this specific channel is free
            CH_SLEEP.stop()                  # Stop lower-priority alarms
            CH_PHONE.stop()
            CH_FACEHIDE.play(alarm_facehide) # Play on dedicated channel
            current_playing = 'facehide'

    # Priority 2: Sleep warning
    elif is_sleepy:
        cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        if not CH_SLEEP.get_busy():
            CH_PHONE.stop()                  # Stop lower-priority alarms
            CH_SLEEP.play(alarm_sleep)        # Play on dedicated channel
            current_playing = 'sleep'

    # Priority 3: Phone warning
    elif phone_detected:
        cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))
        if not CH_PHONE.get_busy():
            CH_PHONE.play(alarm_phone)        # Play on dedicated channel
            current_playing = 'phone'

    # Display active warning overlay if audio is still finishing up
    elif is_audio_busy:
        if current_playing == 'facehide':
            cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        elif current_playing == 'sleep':
            cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        elif current_playing == 'phone':
            cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))

    cv2.imshow("Smart Study Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

