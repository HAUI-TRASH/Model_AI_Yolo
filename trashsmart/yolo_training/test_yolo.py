import argparse

from ultralytics import YOLO
from trashsmart.shared.paths import YOLO_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--save", action="store_true", help="Save prediction image.")
    args = parser.parse_args()

    model = YOLO(str(YOLO_MODEL_PATH))
    results = model(args.image, imgsz=args.imgsz, conf=args.conf, save=args.save)
    results[0].show()


if __name__ == "__main__":
    main()

