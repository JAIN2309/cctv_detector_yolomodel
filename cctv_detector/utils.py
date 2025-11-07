import cv2, os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def blur_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(face, (99, 99), 30)
    return frame

def save_event_image(frame, path):
    cv2.imwrite(path, frame)
