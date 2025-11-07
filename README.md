# 🚨 CCTV Event Detector using YOLOv8

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue">
  <img alt="YOLOv8" src="https://img.shields.io/badge/Model-YOLOv8n-orange">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Enabled-red">
  <img alt="Ultralytics" src="https://img.shields.io/badge/Powered%20by-Ultralytics-yellow">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> 🔍 AI-powered surveillance system detecting desk absence and mobile usage in CCTV feeds using **YOLOv8n**. It saves privacy-protected snapshots and structured event logs for monitoring and analysis.


---

## ✨ Key Features

* **Desk Empty Detection**: Detects when a person is absent from their desk for a configurable duration.
* **Mobile Phone Detection**: Automatically tracks active mobile phone usage.
* **Event Logging**: Saves structured logs in `.csv` and `.json` formats with timestamps, durations, and evidence.
* **Privacy Protection**: Faces in all saved snapshots are automatically blurred using OpenCV.
* **Real-time Preview**: Displays the live feed with bounding boxes and labels.
* **Modular & Scalable**: Easily extendable for multi-camera setups or cloud integration.

---

## 🧰 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.8+ |
| **AI Model** | YOLOv8n (from Ultralytics) |
| **Core Libraries** | OpenCV, NumPy, Ultralytics |
| **Data Handling** | Pandas (for CSV logging) |
| **Outputs** | CSV, JSON, JPG (Snapshots) |

---

## ⚙️ How It Works

The detector uses **YOLOv8n** for high-performance, real-time object detection. It continuously monitors the feed for `"person"` and `"cell phone"` objects and applies custom logic from `detector.py` to identify key events.

**Detection Logic:**

| Event Type | Trigger Condition | Action |
| :--- | :--- | :--- |
| **Desk Empty** | No `person` detected for threshold duration | Logs `desk_empty` event with blurred snapshot |
| **Mobile In Hand** | `cell phone` detected | Logs `mobile_in_hand` event start |
| **Mobile Not In Hand**| `cell phone` disappears after detection | Logs `mobile_not_in_hand` event with duration |

**Workflow:**

1.  **Capture Frame** → Read from CCTV feed, video file, or webcam.
2.  **Object Detection** → `detector.py` uses YOLOv8 to find objects and bounding boxes.
3.  **Event Triggering** → Custom logic tracks object state (e.g., timers for absence).
4.  **Face Anonymization** → `utils.py` finds and blurs faces in frames marked for saving.
5.  **Logging** → `utils.py` saves structured logs and event snapshots to the `outputs/` directory.

---

## 📁 Project Structure

```bash
cctv_detector/
├── main.py              # Core entry point (if using as a module)
├── run_detector.py      # CLI runner for videos or live feeds
├── detector.py          # YOLOv8 detection & event tracking logic
├── utils.py             # Logging, face blurring, image utilities
├── roi.py               # Optional ROI (Region of Interest) handler
├── requirements.txt     # Python dependencies
├── sample_test_video.mp4# Sample video for testing
└── outputs/             # Generated event logs & snapshots
    ├── events.csv
    ├── events.json
    ├── desk_empty_YYYY-MM-DD_HH-MM-SS.jpg
    └── mobile_in_hand_YYYY-MM-DD_HH-MM-SS.jpg
```

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.8 or higher
* Git

### 2. Installation

1️⃣ **Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)<your-username>/cctv_event_detector.git
cd cctv_event_detector
```

2️⃣ **Set up a virtual environment**

<details>
<summary>🪟 Windows (PowerShell)</summary>
```powershell
python -m venv venv
.\venv\Scripts\Activate
```
</details>

<details>
<summary>🐧 macOS / Linux</summary>
```bash
python3 -m venv venv
source venv/bin/activate
```
</details>

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🕹️ Usage

You can run the detector on a pre-recorded video file, a live webcam, or an RTSP stream.

### Run on a Video File

```bash
python run_detector.py --video sample_test_video.mp4
```

### ⚙️ Configuration (Command-Line Arguments)

You can customize the detector's behavior using these optional arguments:

| Argument | Default | Description |
| :--- | :--- | :--- |
| **`--source`** | (Required) | Path to video file, webcam ID (e.g., `0`), or RTSP stream URL. |
| **`--empty-threshold`** | `10.0` | Seconds a person must be absent to trigger a `desk_empty` event. |
| **`--output-dir`** | `outputs/` | Custom folder to save event logs and snapshots. |

---

## 🧾 Sample Output

**JSON Log (`outputs/events.json`):**
```json
{
    "event_type": "mobile_in_hand",
    "timestamp": "2025-11-07_10-25-14",
    "duration_sec": 32.7,
    "image_path": "outputs/mobile_in_hand_2025-11-07_10-25-14.jpg"
}
```

**CSV Log (`outputs/events.csv`):**
```
event_type,timestamp,duration_sec,image_path
desk_empty,2025-11-07_10-22-33,45.2,outputs/desk_empty_2025-11-07_10-22-33.jpg
mobile_in_hand,2025-11-07_10-25-14,32.7,outputs/mobile_in_hand_2025-11-07_10-25-14.jpg
```

---

## 🔐 Privacy Protection

This tool is designed with privacy in mind:

* **Automatic Blurring**: Faces in all saved snapshot images are automatically blurred.
* **Local-First**: No data is uploaded externally. All logs and images stay on your local machine.
* **Full Control**: Logs and snapshots can be deleted or anonymized locally at any time.

---

## 🧭 Future Enhancements

* Multi-camera simultaneous monitoring
* Live web dashboard (using Flask or FastAPI)
* Event notifications (Email, SMS, or Slack)
* Cloud storage integration (AWS S3 / GCP Storage)
* Face recognition-based access control (while respecting privacy)

---
## 👨‍💻 About the Author

<p align="center">
  <strong>Krish Jain</strong>
  <br>
  🎓 L.J. Institute of Computer Applications
  <br>
  💼 <strong>Full Stack Developer & AI Enthusiast</strong>
  <br>
  <br>
  <a href="https://www.linkedin.com/in/krishjain-dev/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"></a>
  <a href="https://github.com/JAIN2309"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"></a>
</p>
---

## 📝 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
