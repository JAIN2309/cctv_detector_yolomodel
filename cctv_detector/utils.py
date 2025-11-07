import cv2, os
import logging # Using logging is better, but we'll stick to print for consistency if you prefer.

# Set up a simple logger for utilities
logger = logging.getLogger(__name__)
# To see these logs, you'd need to configure logging in your main.py
# For simplicity, we can just use print() if you haven't set up logging.

def ensure_dir(path):
    """
    Checks if a directory exists at the given path.
    If it doesn't exist, this function creates it.
    """
    if not os.path.exists(path):
        print(f"📁 [INFO] Directory not found. Creating: {path}")
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"❌ [ERROR] Could not create directory: {path}. Error: {e}")

def blur_faces(frame):
    """
    Finds all faces in a frame and applies a strong blur to them.
    
    NOTE: This function is not very efficient as it re-loads the
    face detection model ('haarcascade_frontalface_default.xml') 
    from disk *every single time* it's called.
    """
    # 1. Convert the frame to grayscale, as face detection works on grayscale images.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Load the pre-trained Haar Cascade model for face detection.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 3. Detect faces in the grayscale image.
    #    - 1.3 = scaleFactor: How much the image size is reduced at each image scale.
    #    - 5   = minNeighbors: How many neighbors each candidate rectangle should have.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 4. Loop over all detected faces
    for (x, y, w, h) in faces:
        # Get the [x, y, width, height] of the face
        
        # 5. Extract the Region of Interest (ROI) - just the face area
        face = frame[y:y+h, x:x+w]
        
        # 6. Apply a very strong Gaussian Blur to the extracted face
        #    - (99, 99) is the kernel size (must be odd). Larger = more blur.
        #    - 30 is the sigma (standard deviation). Larger = more blur.
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        
        # 7. Replace the original face area in the frame with the new blurred face
        frame[y:y+h, x:x+w] = blurred_face
        
    # 8. Return the modified frame
    return frame

def save_event_image(frame, path):
    """
    Saves the given image (frame) to the specified file path.
    """
    try:
        cv2.imwrite(path, frame)
        print(f"💾 [INFO] Snapshot saved: {path}")
    except Exception as e:
        print(f"❌ [ERROR] Failed to save image to {path}. Error: {e}")