import os, time, csv, json, cv2
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from .utils import blur_faces, ensure_dir, save_event_image


class CCTVDetector:
    def __init__(self, video_path, outdir="events", roi=None, empty_threshold=30):
        
        # --- Configuration ---
        self.video_path = video_path  # Path to the video file
        self.outdir = outdir          # Folder to save logs and images
        self.roi = roi                # Region of Interest (optional, not implemented in this logic)
        self.empty_threshold = empty_threshold # Seconds before "desk empty" is triggered
        
        # --- State Tracking ---
        # This is the "memory" of the detector
        self.last_person_time = time.time() # The last timestamp a person was seen
        self.log = []                       # A list to store all event dictionaries
        self.mobile_active = False          # Is a mobile phone currently being held?
        self.mobile_start_time = None       # When did the mobile use start?
        self.desk_empty_logged = False      # Have we already logged the "desk empty" event?

        # --- Setup ---
        ensure_dir(outdir) # Create the output folder if it doesn't exist

        # ✅ Load YOLOv8 model (downloads automatically if missing)
        print("🔄 [INFO] Loading YOLOv8 model (yolov8n.pt)...")
        self.model = YOLO("yolov8n.pt")  # YOLOv8 fast model
        print("✅ [INFO] YOLOv8 model loaded successfully.")

    def run(self):
        # --- 1. Initialize Video Capture ---
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            # Beautified error log
            print(f"❌ [ERROR] Failed to open video file: {self.video_path}")
            return

        print("🚀 [INFO] Starting event detection... (Press 'q' in the window to quit)")

        # --- 2. Start Main Video Loop ---
        while True:
            # Read one frame from the video
            ret, frame = cap.read()
            if not ret:
                # This means the video has ended
                print("ℹ️ [INFO] Video stream ended.")
                break

            # --- 3. Run AI Detection ---
            # 🔍 Run YOLO detection on the current frame
            # verbose=False stops YOLO from printing its own logs every frame
            results = self.model(frame, verbose=False)[0]

            # --- 4. Draw Live Preview ---
            # 🎯 Draw *all* detected bounding boxes on the live frame
            # This loop is for the *live preview window* only
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = self.model.names[int(box.cls)] # Get class name (e.g., "person")
                conf = float(box.conf) # Get confidence score
                
                # Draw the green box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put the label text (e.g., "person 0.85")
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- 5. Check for Events ---
            # These flags will be set based on what was detected *in this frame*
            person_detected = False
            phone_detected = False

            # Check all detections to set our flags
            for box in results.boxes:
                cls_name = self.model.names[int(box.cls)]
                if cls_name == "person":
                    person_detected = True
                # Check for "cell phone" or related terms
                elif cls_name in ["cell phone", "phone", "mobile"]:
                    phone_detected = True

            now = time.time() # Get the current time for logic

            # --- 🧠 Desk Empty Detection Logic ---
            if person_detected:
                # If a person is seen, update the "last seen" time
                self.last_person_time = now
                # Reset the "empty" flag (since they are no longer empty)
                self.desk_empty_logged = False
            else:
                # If no person is seen, check how long they've been gone
                absence_duration = now - self.last_person_time
                
                # If gone longer than our threshold AND we haven't logged it yet
                if absence_duration > self.empty_threshold and not self.desk_empty_logged:
                    print(f"DEBUG: Desk empty threshold breached ({absence_duration:.0f}s)")
                    # Log the event!
                    self.log_event(frame, "desk_empty", int(absence_duration), results)
                    # Set the flag so we don't log this again every frame
                    self.desk_empty_logged = True

            # --- 📱 Mobile in Hand Detection Logic ---
            
            # Event Start: Phone is seen, but wasn't active before
            if phone_detected and not self.mobile_active:
                print("DEBUG: Mobile phone detected, starting timer.")
                self.mobile_active = True
                self.mobile_start_time = now
                # Log the *start* of the event (duration 0)
                self.log_event(frame, "mobile_in_hand", 0, results)
                
            # Event End: Phone is *not* seen, but *was* active
            elif not phone_detected and self.mobile_active:
                print("DEBUG: Mobile phone no longer detected, logging duration.")
                duration = now - self.mobile_start_time
                self.mobile_active = False # Reset the state
                # Log the *end* of the event, with the calculated duration
                self.log_event(frame, "mobile_not_in_hand", round(duration, 2), results)

            # --- 6. Show Live Window ---
            # 👁️ Optional: show live detection window
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nℹ️ [INFO] 'q' key pressed. Stopping...")
                break

        # --- 7. Cleanup ---
        print("ℹ️ [INFO] Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        self.save_log() # Save all logs to files

    def log_event(self, frame, label, duration, results):
        # --- 1. Prepare Log Info ---
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(self.outdir, f"{label}_{ts}.jpg")

        # --- 2. Prepare Snapshot Image ---
        # 🎯 Draw bounding boxes and labels *again*
        # This is because the original 'frame' might be modified by the blur logic
        # Note: This draws on the *same frame* that will be blurred
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 😷 Blur faces before saving final image
        # This modifies the 'frame' variable directly
        frame = blur_faces(frame)

        # 💾 Save the processed frame
        save_event_image(frame, out_path)

        # --- 3. Log Event Details ---
        # 🧾 Log event details to our internal list
        event = {"timestamp": ts, "label": label, "duration": duration}
        self.log.append(event)
        
        # --- Beautified Log Presentation ---
        if label == "desk_empty":
            print(f"🛑 [EVENT] Desk Empty - Person absent for {duration}s")
        elif label == "mobile_in_hand":
            print(f"📱 [EVENT] Mobile Use - Detected @ {ts}")
        elif label == "mobile_not_in_hand":
            print(f"👍 [EVENT] Mobile Use - Ended (Duration: {duration}s)")
        else:
            print(f"🔔 [EVENT] {label} @ {ts} (duration={duration}s)") # Fallback

    def save_log(self):
        # --- 1. Define File Paths ---
        csv_path = os.path.join(self.outdir, "events.csv")
        json_path = os.path.join(self.outdir, "events.json")

        if not self.log:
            print("ℹ️ [INFO] No events were logged. Nothing to save.")
            return

        # --- 2. Save CSV File ---
        print(f"🔄 [INFO] Saving CSV log to {csv_path}...")
        with open(csv_path, "w", newline="") as f:
            # Use the keys from the *first* event as headers
            writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "duration"])
            writer.writeheader()
            writer.writerows(self.log)
            
        # --- 3. Save JSON File ---
        print(f"🔄 [INFO] Saving JSON log to {json_path}...")
        with open(json_path, "w") as f:
            json.dump(self.log, f, indent=4) # indent=4 makes it human-readable
            
        # --- Beautified Final Log ---
        print("\n" + "="*30)
        print("✅ [SUCCESS] All logs saved!")
        print(f"  - CSV:  {csv_path}")
        print(f"  - JSON: {json_path}")
        print("="*30 + "\n")