import os, time, csv, json, cv2
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from .utils import blur_faces, ensure_dir, save_event_image


class CCTVDetector:
    def __init__(self, video_path, outdir="events", roi=None, empty_threshold=30):
        self.video_path = video_path
        self.outdir = outdir
        ensure_dir(outdir)
        self.roi = roi
        self.empty_threshold = empty_threshold
        self.last_person_time = time.time()
        self.log = []

        # State tracking
        self.mobile_active = False
        self.mobile_start_time = None
        self.desk_empty_logged = False

        # ✅ Load YOLOv8 model (downloads automatically if missing)
        print("🔄 Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")  # YOLOv8 fast model
        print("✅ YOLOv8 model ready.")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ Error opening video:", self.video_path)
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 🔍 Run YOLO detection
            results = self.model(frame, verbose=False)[0]

            # 🎯 Draw bounding boxes before further logic
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = self.model.names[int(box.cls)]
                conf = float(box.conf)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            person_detected = False
            phone_detected = False

            for box in results.boxes:
                cls_name = self.model.names[int(box.cls)]
                if cls_name == "person":
                    person_detected = True
                elif cls_name in ["cell phone", "phone", "mobile"]:
                    phone_detected = True

            now = time.time()

            # 🧠 Desk empty detection logic
            if person_detected:
                self.last_person_time = now
                self.desk_empty_logged = False
            else:
                if now - self.last_person_time > self.empty_threshold and not self.desk_empty_logged:
                    duration = int(now - self.last_person_time)
                    self.log_event(frame, "desk_empty", duration, results)
                    self.desk_empty_logged = True

            # 📱 Mobile in hand detection logic
            if phone_detected and not self.mobile_active:
                self.mobile_active = True
                self.mobile_start_time = now
                self.log_event(frame, "mobile_in_hand", 0, results)
            elif not phone_detected and self.mobile_active:
                duration = now - self.mobile_start_time
                self.mobile_active = False
                self.log_event(frame, "mobile_not_in_hand", round(duration, 2), results)

            # 👁️ Optional: show live detection window
            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.save_log()

    def log_event(self, frame, label, duration, results):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = os.path.join(self.outdir, f"{label}_{ts}.jpg")

        # 🎯 Draw bounding boxes and labels
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 😷 Blur faces before saving final image
        frame = blur_faces(frame)

        # 💾 Save the processed frame
        save_event_image(frame, out_path)

        # 🧾 Log event details
        event = {"timestamp": ts, "label": label, "duration": duration}
        self.log.append(event)
        print(f"[EVENT] {label} @ {ts} (duration={duration}s)")

    def save_log(self):
        csv_path = os.path.join(self.outdir, "events.csv")
        json_path = os.path.join(self.outdir, "events.json")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "label", "duration"])
            writer.writeheader()
            writer.writerows(self.log)
        with open(json_path, "w") as f:
            json.dump(self.log, f, indent=4)
        print(f"\n✅ Logs saved to:\n  {csv_path}\n  {json_path}")
