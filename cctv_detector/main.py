import argparse
from .detector import CCTVDetector

def main():
    parser = argparse.ArgumentParser(description="CCTV Event Detector")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--roi", nargs=4, type=int, default=None, help="x y w h (optional)")
    parser.add_argument("--empty-threshold", type=int, default=8, help="Seconds before empty desk triggers")
    parser.add_argument("--outdir", default="events", help="Output folder")
    args = parser.parse_args()

    detector = CCTVDetector(video_path=args.video, outdir=args.outdir,
                            roi=args.roi, empty_threshold=args.empty_threshold)
    detector.run()

if __name__ == "__main__":
    main()
