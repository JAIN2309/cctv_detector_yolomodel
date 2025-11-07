# 🚨 CCTV Event Detector using YOLOv8  

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-red)
![Ultralytics](https://img.shields.io/badge/Powered%20by-Ultralytics-yellow)

> 🔍 **AI-powered surveillance system** detecting desk absence and mobile usage in CCTV feeds using **YOLOv8n**, saving privacy-protected snapshots and structured logs for monitoring and analysis.

---

## ✨ Key Features  

- **Desk Empty Detection** – Detect when a person is absent for a configurable duration.  
- **Mobile Phone Detection** – Track mobile phone usage automatically.  
- **Event Logging** – Logs in `.csv` and `.json` formats with timestamps and durations.  
- **Privacy Protection** – Automatically blurs faces in snapshots.  
- **Real-time Preview** – Shows live bounding boxes with object labels.  
- **Modular & Scalable** – Designed for easy extension, including multi-camera support or cloud integration.  

---

## ⚙️ How It Works  

The system leverages **YOLOv8n** for real-time object detection on video frames. It tracks objects `"person"` and `"cell phone"` and applies event-triggering logic.

### 🧠 Detection Logic  

| Event Type | Trigger Condition | Action |
|------------|-----------------|--------|
| **Desk Empty** | No person detected for threshold time | Logs `desk_empty` with blurred snapshot |
| **Mobile In Hand** | Cell phone detected | Logs `mobile_in_hand` event start |
| **Mobile Not In Hand** | Cell phone disappears | Logs `mobile_not_in_hand` event with duration |

### 🧩 Workflow  

1. **Capture Frame** → Read from CCTV feed or video file  
2. **Object Detection** → YOLOv8 detects bounding boxes  
3. **Event Triggering** → Detect state changes  
4. **Face Anonymization** → Blur faces using OpenCV  
5. **Logging** → Save structured data + event snapshot  

---

## 📁 Project Structure  

```bash
cctv_detector/
├── main.py              # Core entry point
├── run_detector.py      # CLI runner for videos or live feeds
├── detector.py          # YOLOv8 detection & event logic
├── utils.py             # Logging, face blurring, image utilities
├── roi.py               # Optional ROI (Region of Interest) handler
├── requirements.txt     # Python dependencies
├── sample_test_video.mp4 # Sample video for testing
├── outputs/             # Generated event logs & snapshots
│   ├── events.csv
│   ├── events.json
│   ├── desk_empty_YYYY-MM-DD_HH-MM-SS.jpg
│   └── mobile_in_hand_YYYY-MM-DD_HH-MM-SS.jpg

🚀 Installation & Usage
1️⃣ Clone the repository
git clone https://github.com/<your-username>/cctv_event_detector.git
cd cctv_event_detector

2️⃣ Set up virtual environment
<details> <summary>🪟 Windows (PowerShell)</summary>
python -m venv venv
venv\Scripts\activate

</details> <details> <summary>🐧 macOS / Linux</summary>
python3 -m venv venv
source venv/bin/activate

</details>
3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the detector
python run_detector.py --video sample_test_video.mp4


Optional arguments:

--empty-threshold <seconds> → Set desk absence duration threshold

--output-dir <folder> → Custom folder for logs and snapshots

🧾 Sample Output
JSON Log
{
    "event_type": "mobile_in_hand",
    "timestamp": "2025-11-07_10-25-14",
    "duration_sec": 32.7,
    "image_path": "outputs/mobile_in_hand_2025-11-07_10-25-14.jpg"
}

CSV Log
event_type,timestamp,duration_sec,image_path
desk_empty,2025-11-07_10-22-33,45.2,outputs/desk_empty_2025-11-07_10-22-33.jpg
mobile_in_hand,2025-11-07_10-25-14,32.7,outputs/mobile_in_hand_2025-11-07_10-25-14.jpg

🔐 Privacy Protection

All saved frames have faces automatically blurred.

No data is uploaded externally.

Logs can be deleted or anonymized locally anytime.

🧭 Future Enhancements

Multi-camera simultaneous monitoring

Web dashboard (Flask/Django) with live analytics

Event notifications via Email/SMS

Cloud storage integration (AWS/GCP)

Face recognition-based access control

🧰 Tech Stack
Component	Technology
Language	Python 3.8+
AI Model	YOLOv8n (Ultralytics)
Libraries	OpenCV, NumPy, Pandas, Ultralytics
Outputs	CSV, JSON, JPG
👨‍💻 Author

Krish Jain
🎓 L.J. Institute of Computer Applications
💼 Full Stack & Computer Vision Developer
🌐 GitHub
 • LinkedIn

📝 License

This project is licensed under MIT License. See LICENSE
 for details.
