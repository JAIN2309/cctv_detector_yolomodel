import os, time, csv, json, cv2
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from .utils import blur_faces, ensure_dir, save_event_image


class CCTVDetector:
    # --- [PERFECTED] ---
    # We add constants for drawing to make it easy to change colors
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255) # White
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_THICKNESS = 1
    # --- [END PERFECTED] ---

    def __init__(self, video_path, outdir="events", roi=None, empty_threshold=30):
        
        # --- Configuration ---
        self.video_path = video_path
        self.outdir = outdir
        self.roi = roi
        self.empty_threshold = empty_threshold
        
        # --- State Tracking ---
        self.last_person_time = time.time()
        self.log = []
        self.mobile_active = False
        self.mobile_start_time = None
        self.desk_empty_logged = False

        # --- Setup ---
        ensure_dir(outdir)
        print("🔄 [INFO] Loading YOLOv8 model (yolov8n.pt)...")
        self.model = YOLO("yolov8n.pt")
        print("✅ [INFO] YOLOv8 model loaded successfully.")

    # --- [PERFECTED] ---
    # We create ONE function to handle all drawing.
    # This fixes the bug and makes the labels look professional.
    def _draw_detections(self, frame, results):
        """
        Draws professional-looking bounding boxes and labels on a *copy* of the frame.
        """
        # Work on a copy so we never modify the original frame
        output_frame = frame.copy() 
        
        for box in results.boxes:
            # Get coordinates, class, and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls)
            cls_name = self.model.names.get(cls_id, "Unknown")
            conf = float(box.conf)
            label = f"{cls_name} {conf:.2f}"
            
            # --- Professional Label ---
            # 1. Get the size of the text label
            (w, h), _ = cv2.getTextSize(label, self.FONT, self.FONT_SCALE, self.FONT_THICKNESS)
            
            # 2. Calculate position for the label background
            #    This puts it just above the box, or just below if at the top edge
            label_y = y1 - 10 if y1 - 10 > h else y1 + h + 10
            
            # 3. Draw the filled background rectangle
            cv2.rectangle(output_frame, (x1, label_y - h - 5), (x1 + w, label_y), 
                          self.BOX_COLOR, -1) # -1 thickness = filled
            
            # 4. Draw the white text on top of the background
            cv2.putText(output_frame, label, (x1, label_y - 3), 
                        self.FONT, self.FONT_SCALE, self.TEXT_COLOR, 
                        self.FONT_THICKNESS, cv2.LINE_AA)
            
            # 5. Draw the bounding box (outline)
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), self.BOX_COLOR, self.FONT_THICKNESS + 1)
            
        return output_frame
    # --- [END PERFECTED] ---

    def run(self):
        # --- 1. Initialize Video Capture ---
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ [ERROR] Failed to open video file: {self.video_path}")
            return

        print("🚀 [INFO] Starting event detection... (Press 'q' in the window to quit)")

        # --- 2. Start Main Video Loop ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ℹ️ [INFO] Video stream ended.")
                break

            # --- 3. Run AI Detection ---
            results = self.model(frame, verbose=False)[0]

            # --- 4. Draw Live Preview ---
            # --- [PERFECTED] ---
            # We no longer draw here. We call our new function.
            # This 'preview_frame' has the perfected labels.
            preview_frame = self._draw_detections(frame, results)
            # --- [END PERFECTED] ---

            # --- 5. Check for Events ---
            person_detected = False
            phone_detected = False

            for box in results.boxes:
                cls_name = self.model.names[int(box.cls)]
                if cls_name == "person":
                    person_detected = True
                elif cls_name in ["cell phone", "phone", "mobile"]:
                    phone_detected = True

            now = time.time()

            # --- 🧠 Desk Empty Detection Logic ---
            if person_detected:
                self.last_person_time = now
                self.desk_empty_logged = False
            else:
                absence_duration = now - self.last_person_time
                if absence_duration > self.empty_threshold and not self.desk_empty_logged:
                    print(f"DEBUG: Desk empty threshold breached ({absence_duration:.0f}s)")
                    # --- [PERFECTED] ---
                    # We pass the ORIGINAL 'frame', not the preview_frame,
                    # to log_event. This is cleaner.
                    self.log_event(frame, "desk_empty", int(absence_duration), results)
                    # --- [END PERFECTED] ---
                    self.desk_empty_logged = True

            # --- 📱 Mobile in Hand Detection Logic ---
            if phone_detected and not self.mobile_active:
                print("DEBUG: Mobile phone detected, starting timer.")
                self.mobile_active = True
                self.mobile_start_time = now
                self.log_event(frame, "mobile_in_hand", 0, results)
                
            elif not phone_detected and self.mobile_active:
                print("DEBUG: Mobile phone no longer detected, logging duration.")
                duration = now - self.mobile_start_time
                self.mobile_active = False
                self.log_event(frame, "mobile_not_in_hand", round(duration, 2), results)

            # --- 6. Show Live Window ---
            # --- [PERFECTED] ---
            # We show the 'preview_frame' which has the perfected labels
            cv2.imshow("YOLOv8 Detection", preview_frame)
            # --- [END PERFECTED] ---
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nℹ️ [INFO] 'q' key pressed. Stopping...")
                break

        # --- 7. Cleanup ---
        print("ℹ️ [INFO] Cleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        self.save_log()

    def log_event(self, frame, label, duration, results):
        # --- 1. Prepare Log Info ---
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(self.outdir, f"{label}_{ts}.jpg")

        # --- 2. Prepare Snapshot Image ---
        # --- [PERFECTED] ---
        # We call our new drawing function to create the snapshot
        # We pass frame.copy() to be 100% sure we don't cause bugs
        annotated_frame = self._draw_detections(frame.copy(), results)
        # --- [END PERFECTED] ---

        # 😷 Blur faces before saving final image
        # This modifies the 'annotated_frame' variable
        blurred_frame = blur_faces(annotated_frame)

        # 💾 Save the processed frame
        save_event_image(blurred_frame, out_path) # Save the blurred frame

        # --- 3. Log Event Details ---
        # --- [PERFECTED] ---
        # I added "image_path" to your log, which your README file
        # said it should have. This makes your logs complete.
        event = {
            "timestamp": ts, 
            "label": label, 
            "duration": duration, 
            "image_path": out_path
        }
        # --- [END PERFECTED] ---
        self.log.append(event)
        
        # --- Beautified Log Presentation ---
        if label == "desk_empty":
            print(f"🛑 [EVENT] Desk Empty - Person absent for {duration}s")
        elif label == "mobile_in_hand":
            print(f"📱 [EVENT] Mobile Use - Detected @ {ts}")
        elif label == "mobile_not_in_hand":
            print(f"👍 [EVENT] Mobile Use - Ended (Duration: {duration}s)")
        else:
            print(f"🔔 [EVENT] {label} @ {ts} (duration={duration}s)")

    def save_log(self):
        csv_path = os.path.join(self.outdir, "events.csv")
        json_path = os.path.join(self.outdir, "events.json")

        if not self.log:
            print("ℹ️ [INFO] No events were logged. Nothing to save.")
            return

        # --- [PERFECTED] ---
        # This code is now more robust. It gets the headers
        # from the first log entry, so it will work even if
        # you add more fields to your 'event' dictionary.
        try:
            fieldnames = self.log[0].keys()
        except IndexError:
            print("ℹ️ [INFO] No events in log, nothing to save.")
            return
            
        print(f"🔄 [INFO] Saving CSV log to {csv_path}...")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # --- [END PERFECTED] ---
            writer.writeheader()
            writer.writerows(self.log)
            
        print(f"🔄 [INFO] Saving JSON log to {json_path}...")
        with open(json_path, "w") as f:
            json.dump(self.log, f, indent=4)
            
        print("\n" + "="*30)
        print("✅ [SUCCESS] All logs saved!")
        print(f" 	- CSV: 	{csv_path}")
        print(f" 	- JSON: {json_path}")
        print("="*30 + "\n")