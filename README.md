Ah! I understand now — you want **one complete, ready-to-use `README.md` file**, without fragmented sections or extra separations, fully formatted and perfectly aligned. I will rewrite it in a **single, cohesive markdown file**, ready to save and use directly in your GitHub repo. Here's the complete version:

````markdown
# 🚨 CCTV Event Detector using YOLOv8  

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8n-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-red)
![Ultralytics](https://img.shields.io/badge/Powered%20by-Ultralytics-yellow)

> 🔍 AI-powered surveillance system detecting desk absence and mobile usage in CCTV feeds using **YOLOv8n**, saving privacy-protected snapshots and structured logs for monitoring and analysis.

---

## ✨ Key Features  

- **Desk Empty Detection** – Detects when a person is absent for a configurable duration.  
- **Mobile Phone Detection** – Automatically tracks mobile phone usage.  
- **Event Logging** – Saves logs in `.csv` and `.json` formats with timestamps and durations.  
- **Privacy Protection** – Faces in snapshots are automatically blurred.  
- **Real-time Preview** – Displays live bounding boxes with object labels.  
- **Modular & Scalable** – Easily extendable for multi-camera setups or cloud integration.  

---

## ⚙️ How It Works  

The detector uses **YOLOv8n** for real-time object detection. It continuously monitors `"person"` and `"cell phone"` objects and applies custom logic to detect key events.

**Detection Logic:**  

| Event Type | Trigger Condition | Action |
|------------|-----------------|--------|
| Desk Empty | No person detected for threshold duration | Logs `desk_empty` event with blurred snapshot |
| Mobile In Hand | Cell phone detected | Logs `mobile_in_hand` event start |
| Mobile Not In Hand | Cell phone disappears | Logs `mobile_not_in_hand` event with duration |

**Workflow:**  
1. Capture Frame → Read from CCTV feed or video file  
2. Object Detection → YOLOv8 detects objects and bounding boxes  
3. Event Triggering → Detect state changes  
4. Face Anonymization → Blur faces using OpenCV  
5. Logging → Save structured logs and event snapshots  

---

## 📁 Project Structure  

```bash
cctv_detector/
├── main.py                # Core entry point
├── run_detector.py        # CLI runner for videos or live feeds
├── detector.py            # YOLOv8 detection & event logic
├── utils.py               # Logging, face blurring, image utilities
├── roi.py                 # Optional ROI (Region of Interest) handler
├── requirements.txt       # Python dependencies
├── sample_test_video.mp4  # Sample video for testing
├── outputs/               # Generated event logs & snapshots
│   ├── events.csv
│   ├── events.json
│   ├── desk_empty_YYYY-MM-DD_HH-MM-SS.jpg
│   └── mobile_in_hand_YYYY-MM-DD_HH-MM-SS.jpg
````

---

## 🚀 Installation & Usage

1️⃣ **Clone the repository**

```bash
git clone https://github.com/<your-username>/cctv_event_detector.git
cd cctv_event_detector
```

2️⃣ **Set up virtual environment**

<details>
<summary>🪟 Windows (PowerShell)</summary>
```powershell
python -m venv venv
venv\Scripts\activate
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

4️⃣ **Run the detector**

```bash
python run_detector.py --video sample_test_video.mp4
```

**Optional arguments:**

* `--empty-threshold <seconds>` → Set desk absence duration threshold
* `--output-dir <folder>` → Custom folder for logs and snapshots

---

## 🧾 Sample Output

**JSON Log:**

```json
{
    "event_type": "mobile_in_hand",
    "timestamp": "2025-11-07_10-25-14",
    "duration_sec": 32.7,
    "image_path": "outputs/mobile_in_hand_2025-11-07_10-25-14.jpg"
}
```

**CSV Log:**

```
event_type,timestamp,duration_sec,image_path
desk_empty,2025-11-07_10-22-33,45.2,outputs/desk_empty_2025-11-07_10-22-33.jpg
mobile_in_hand,2025-11-07_10-25-14,32.7,outputs/mobile_in_hand_2025-11-07_10-25-14.jpg
```

---

## 🔐 Privacy Protection

* Faces in all saved frames are automatically blurred.
* No data is uploaded externally.
* Logs can be deleted or anonymized locally anytime.

---

## 🧭 Future Enhancements

* Multi-camera simultaneous monitoring
* Live web dashboard (Flask/Django)
* Event notifications via Email/SMS
* Cloud storage integration (AWS/GCP)
* Face recognition-based access control

---

## 🧰 Tech Stack

| Component | Technology                         |
| --------- | ---------------------------------- |
| Language  | Python 3.8+                        |
| AI Model  | YOLOv8n (Ultralytics)              |
| Libraries | OpenCV, NumPy, Pandas, Ultralytics |
| Outputs   | CSV, JSON, JPG                     |

---

## 👨‍💻 Author

**Krish Jain**
🎓 L.J. Institute of Computer Applications
💼 Full Stack & Computer Vision Developer
🌐 [GitHub]([https://github.com/](https://github.com/JAIN2309)) • [LinkedIn]([https://www.linkedin.com/in/](https://www.linkedin.com/in/krishjain-dev/))

---

## 📝 License

This project is licensed under **MIT License**. See [LICENSE](LICENSE) for details.

```

.  

Do you want me to do that next?
```
