"""Classical RGB parcel candidates. Colors are hypotheses, not object truth."""
import cv2
import numpy as np


def rgb_image(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("Expected finite RGB image in HWC or [1,]CHW layout")
    if np.issubdtype(image.dtype, np.floating):
        if image.min() < 0 or image.max() > 1.001:
            raise ValueError("Vision must receive raw [0,1] images before policy normalization")
        image = image * 255
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


class ParcelDetector:
    def __init__(self, config):
        self.config = config
        self.previous = {}
        self.counts = {}
        x0, y0, x1, y1 = config["roi"]
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError("roi must be normalized x0,y0,x1,y1")
        if not 0 < config["min_area_fraction"] < config["max_area_fraction"] <= 1:
            raise ValueError("Invalid candidate area fraction bounds")
        if config["stable_frames"] < 1 or config["max_center_jump_px"] <= 0:
            raise ValueError("Tracking thresholds must be positive")

    def detect(self, rgb):
        image = rgb_image(rgb)
        height, width = image.shape[:2]
        x0, y0, x1, y1 = self.config["roi"]
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError("roi must be normalized x0,y0,x1,y1")
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[int(y0 * height):int(y1 * height), int(x0 * width):int(x1 * width)] = 255
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        kernel = np.ones((3, 3), np.uint8)
        # Segment a whole parcel before classifying it. White shipping labels
        # otherwise fragment a body into tape-colored strips and small patches.
        color_masks = {label: cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
                       for label, (lower, upper) in self.config["targets"].items()}
        # Assign overlaps to narrower/more saturated classes before brown.
        occupied = np.zeros((height, width), np.uint8)
        exclusive = {}
        for label in [key for key in ("orange", "yellow", "blue", "brown") if key in color_masks] + [
            key for key in color_masks if key not in ("orange", "yellow", "blue", "brown")]:
            exclusive[label] = cv2.bitwise_and(color_masks[label], cv2.bitwise_not(occupied))
            occupied = cv2.bitwise_or(occupied, color_masks[label])
        mask = occupied
        if self.config.get("include_bright_labels", False):
            labels = cv2.inRange(hsv, np.array([0, 0, self.config["label_min_value"]], np.uint8),
                                 np.array([179, self.config["label_max_saturation"], 255], np.uint8))
            mask = cv2.bitwise_or(mask, labels)
        if self.config.get("table_segmentation", False):
            # Use the observed green tabletop as foreground support. This keeps
            # low-saturation parcel bodies separate from the wall and floor,
            # instead of joining nearby boxes through broad color thresholds.
            lower, upper = self.config["table_hsv"]
            green = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
            green = cv2.bitwise_and(green, roi)
            surfaces, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            table_support = np.zeros_like(green)
            if surfaces:
                table = max(surfaces, key=cv2.contourArea)
                if cv2.contourArea(table) > height * width * 0.05:
                    cv2.drawContours(table_support, [cv2.convexHull(table)], -1, 255, cv2.FILLED)
            # No visible supporting tabletop -> report no candidates, rather
            # than treating an unrelated background blob as a parcel.
            mask = cv2.bitwise_and(table_support, cv2.bitwise_not(green))
        mask = cv2.bitwise_and(mask, roi)
        # Opening removes narrow bridges; closing can merge neighboring boxes.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        grouped = {label: [] for label in color_masks}
        for contour in contours:
                area = cv2.contourArea(contour)
                fraction = area / (height * width)
                if not self.config["min_area_fraction"] <= fraction <= self.config["max_area_fraction"]:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if max(w / h, h / w) > self.config.get("max_aspect_ratio", 4):
                    continue
                fill = area / (w * h)
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = area / hull_area if hull_area else 0.0
                if fill < self.config["min_fill_ratio"] or solidity < self.config["min_solidity"]:
                    continue
                moments = cv2.moments(contour)
                uv = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
                region = np.zeros((height, width), np.uint8)
                cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
                support = {label: cv2.countNonZero(cv2.bitwise_and(region, cmask))
                           for label, cmask in exclusive.items()}
                label = max(support, key=support.get)
                color_fraction = support[label] / max(1, cv2.countNonZero(region))
                if color_fraction < self.config.get("min_color_fraction", 0.08):
                    if self.config.get("table_segmentation", False):
                        label = "unknown"
                        grouped.setdefault(label, [])
                    else:
                        continue
                clipped = x <= int(x0 * width) or y <= int(y0 * height) or (
                    x + w >= int(x1 * width) or y + h >= int(y1 * height))
                grouped[label].append((area, uv, [x, y, w, h], fill, solidity, clipped, color_fraction))
        objects = []
        for label, candidates in grouped.items():
            if not candidates:
                self.previous.pop(label, None)
                self.counts[label] = 0
                continue
            previous = self.previous.get(label)
            # Track the nearest candidate; reacquire the largest if it disappeared.
            if previous is not None:
                best = min(candidates, key=lambda item: np.linalg.norm(item[1] - previous))
                matched = np.linalg.norm(best[1] - previous) <= self.config["max_center_jump_px"]
                if not matched:
                    best = max(candidates, key=lambda item: item[0])
            else:
                best, matched = max(candidates, key=lambda item: item[0]), False
            self.counts[label] = self.counts.get(label, 0) + 1 if matched else 1
            self.previous[label] = best[1]
            objects.append({
                "label": label, "uv": best[1].tolist(), "bbox": best[2],
                "area_px": float(best[0]), "fill": float(best[3]),
                "solidity": float(best[4]), "candidate_count": len(candidates),
                "clipped": bool(best[5]), "color_fraction": float(best[6]),
                "stable": self.counts[label] >= self.config["stable_frames"],
            })
        return objects

    @staticmethod
    def annotate(rgb, objects):
        image = cv2.cvtColor(rgb_image(rgb), cv2.COLOR_RGB2BGR)
        for item in objects:
            x, y, w, h = item["bbox"]
            color = (0, 220, 0) if item["stable"] else (0, 190, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.circle(image, tuple(round(v) for v in item["uv"]), 3, color, -1)
            tag = " clipped" if item.get("clipped") else ""
            cv2.putText(image, f'{item["label"]} ({item["candidate_count"]}){tag}',
                        (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return image
