# 🚨 CCTV Event Detector using YOLOv8  

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-red)
![Ultralytics](https://img.shields.io/badge/Powered%20by-Ultralytics-yellow)

> 🔍 A **Python-based AI surveillance tool** that detects key events (desk absence & mobile usage) from CCTV feeds using the **YOLOv8** object detection model.  
> Logs events with timestamps, durations, and privacy-blurred snapshots for secure monitoring and analysis.

---

## ✨ Features  

✅ **Desk Empty Detection** – Detects and logs when a person has been absent from the camera view for a specified duration.  
✅ **Mobile Phone Detection** – Identifies and tracks mobile phone usage events automatically.  
✅ **Event Logging** – Stores all detected events in both `.csv` and `.json` with precise timestamps and durations.  
✅ **Snapshot Archiving** – Saves an image for each event with **blurred faces** to preserve privacy.  
✅ **Live Video Preview** – Displays real-time bounding boxes with object labels during processing.  
✅ **Modular & Scalable** – Built with reusable modules for future AI/ML and cloud integrations.

---

## ⚙️ How It Works  

The detector uses **Ultralytics YOLOv8n** to process video frames in real-time.  
It continuously monitors for **"person"** and **"cell phone"** objects, applying custom logic to detect behavioral events.

### 🧠 Detection Logic Overview  

| Event Type | Trigger Condition | Action Taken |
|-------------|------------------|---------------|
| **Desk Empty** | No `person` detected beyond threshold | Logs `desk_empty` event with snapshot |
| **Mobile In Hand** | `cell phone` detected | Logs `mobile_in_hand` event start |
| **Mobile Not In Hand** | `cell phone` disappears | Logs `mobile_not_in_hand` event with duration |

### 🧩 Event Workflow  

1. **Frame Read** → Capture frame from camera or video file  
2. **Object Detection** → YOLOv8 detects bounding boxes  
3. **Event Trigger** → Logic determines state change  
4. **Face Blur** → Automatically anonymizes faces in the frame  
5. **Log & Save** → Saves structured metadata + event image  

---

## 📁 Folder Structure  

```bash
cctv_detector/
├── main.py              # Core entry point
├── detector.py          # YOLOv8 detection and event logic
├── utils.py             # Utilities for logging & blurring
├── roi.py               # Optional ROI handling
run_detector.py          # Command-line runner
requirements.txt         # Python dependencies
sample_test_video.mp4    # Example video for testing

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/cctv_event_detector.git
cd cctv_event_detector

2️⃣ Set Up Virtual Environment
<details> <summary>🪟 Windows (PowerShell)</summary>
python -m venv venv
venv\Scripts\activate

</details> <details> <summary>🐧 macOS / Linux</summary>
python3 -m venv venv
source venv/bin/activate

</details>
3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Detector
python run_detector.py --video sample_test_video.mp4


💡 Optional arguments:
--empty-threshold → Set desk empty duration threshold (seconds)
--output-dir → Specify custom output folder

🧠 Tech Stack
Component	Technology
Language	Python 3.8+
AI Model	YOLOv8n (Ultralytics)
Libraries	OpenCV, NumPy, Pandas, Ultralytics
Outputs	CSV, JSON, JPG
📊 Example Outputs

Generated Files:

outputs/
├── events.csv
├── events.json
├── desk_empty_2025-11-07_10-22-33.jpg
├── mobile_in_hand_2025-11-07_10-25-14.jpg


Sample Log Entry (JSON):

{
    "event_type": "mobile_in_hand",
    "timestamp": "2025-11-07_10-25-14",
    "duration_sec": 32.7,
    "image_path": "outputs/mobile_in_hand_2025-11-07_10-25-14.jpg"
}

🔐 Privacy Protection

✅ All saved frames undergo automatic face blurring using OpenCV.

✅ No data is uploaded or shared externally.

✅ Local logs can be deleted or anonymized anytime.

🧭 Future Enhancements

 Multi-camera concurrent monitoring

 Flask/Django-based live web dashboard

 Email/SMS event notifications

 Integration with cloud storage (AWS/GCP)

 Face recognition-based access control

🧾 License

This project is licensed under the MIT License.
See the LICENSE
 file for more information.

👨‍💻 Author

Krish Jain
🎓 L.J. Institute of Computer Applications
💼 Full Stack & Computer Vision Developer
🌐 GitHub
 • LinkedIn
