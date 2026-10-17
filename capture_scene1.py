#!/usr/bin/env python3
"""Read original scene1 sensor topics and save samples; send no controls."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from scene1.vision import ParcelDetector

ROOT = Path(__file__).resolve().parent


def main():
    import rospy
    from sensor_msgs.msg import CompressedImage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "captures" / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    rospy.init_node("scene1_vision_capture", anonymous=True)
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, topic in (("head", "/cam_h/color/image_raw/compressed"),
                        ("left", "/cam_l/color/image_raw/compressed"),
                        ("right", "/cam_r/color/image_raw/compressed")):
        detector = ParcelDetector(config["vision"] if name == "head" else {
            **config["vision"], "roi": [0, 0, 1, 1], "table_segmentation": False})
        last_stamp = None
        for _ in range(config["vision"]["stable_frames"]):
            deadline = time.monotonic() + args.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"No advancing camera frame on {topic}")
                message = rospy.wait_for_message(topic, CompressedImage, timeout=remaining)
                stamp = message.header.stamp.to_sec()
                if stamp != last_stamp:
                    last_stamp = stamp
                    break
            bgr = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode {topic}")
            bgr = cv2.resize(bgr, (848, 480))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            detections = detector.detect(rgb)
        cv2.imwrite(str(args.output / f"{name}-raw.png"), bgr)
        cv2.imwrite(str(args.output / f"{name}-annotated.jpg"), detector.annotate(rgb, detections))
        results[name] = {"topic": topic, "stamp": stamp, "image_wh": [848, 480], "detections": detections}
    (args.output / "detections.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "sensors_only": True, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
