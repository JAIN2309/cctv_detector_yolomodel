import argparse
# Import the main detector class from its file within this package
from .detector import CCTVDetector

def main():
    print("🚀 [INFO] Initializing CCTV Event Detector...")
    
    # --- 1. Set up Argument Parser ---
    # This creates a tool to read commands from the terminal
    parser = argparse.ArgumentParser(description="CCTV Event Detector")

    # --- 2. Define Command-Line Arguments ---
    # The user *must* provide a video source
    parser.add_argument("--video", required=True, help="Path to video file")
    
    # Optional: Specify a Region of Interest (ROI)
    parser.add_argument("--roi", nargs=4, type=int, default=None, 
                        help="Defines a detection zone: x y w h (optional)")
    
    # Optional: Set the absence threshold
    parser.add_argument("--empty-threshold", type=int, default=8, 
                        help="Seconds before 'desk empty' event triggers (default: 8)")
    
    # Optional: Specify the output folder
    parser.add_argument("--outdir", default="events", 
                        help="Folder to save logs and snapshots (default: 'events')")

    # --- 3. Parse the Arguments ---
    # Read the arguments provided by the user in the terminal
    args = parser.parse_args()

    # --- 4. Create and Run the Detector ---
    # Create an instance of our detector class, passing in all the user's settings
    print("ℹ️ [INFO] Configuration loaded. Starting detector...")
    detector = CCTVDetector(video_path=args.video, 
                            outdir=args.outdir,
                            roi=args.roi, 
                            empty_threshold=args.empty_threshold)
    
    # Start the main detection loop
    detector.run()

# --- 5. Script Entry Point ---
# This standard Python check allows this file to be run directly
# e.g., using `python -m cctv_detector.main --video ...`
if __name__ == "__main__":
    main()