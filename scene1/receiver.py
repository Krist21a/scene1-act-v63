"""Visible near-face centering for the receiving hand, without scene truth."""
import cv2
import numpy as np


class NearFaceProjector:
    """Keep the original roof validation and require the visible closing section.

    Used only around subsequent parcels' local weighing/regrasp calls. The
    original projector, grasp state machine and collision checks stay intact.
    """
    def __init__(self, projector):
        self.projector = projector

    def __getattr__(self, name):
        return getattr(self.projector, name)

    def measure_outline(self, side, target, measured, rgb):
        original = self.projector.measure_outline(side, target, measured, rgb)
        if original is None or side != 1:
            return original
        major = np.asarray(target.get('major_axis_xy', [1., 0.]), float)
        major = major/max(np.linalg.norm(major), 1e-9)
        minor = np.array([-major[1], major[0]])
        extent = np.asarray(target.get('oriented_extent_m', target['extent_xy_m']))
        # Settlement records the observed support height on the regrasp target.
        plane = target.get('support_z_base_m')
        if plane is None:
            return None
        center = np.asarray(target['position_base_m'])
        corners = [np.r_[center[:2]+a*extent[0]/2*major+b*extent[1]/2*minor, z]
                   for a in (-1, 1) for b in (-1, 1)
                   for z in (plane, target['top_z_base_m'])]
        h, w = rgb.shape[:2]
        projected = [self.projector.project(side, p, measured, [w, h], allow_outside=True)
                     for p in corners]
        if any(p is None for p in projected):
            return None
        hull = cv2.convexHull(np.asarray([p['uv'] for p in projected], np.float32)).astype(np.int32)
        tool = self.projector.kinematics.matrix(np.asarray(measured)[8:15], side)
        closing = self.projector.project(side, tool[:3, 3], measured, [w, h], allow_outside=True)
        roof = center.copy(); roof[2] = target['top_z_base_m']
        reference = self.projector.project(side, roof, measured, [w, h], allow_outside=True)
        if closing is None or reference is None:
            return None
        section = receiver_section(rgb, hull, closing['uv'])
        if section is None:
            return None
        correction = reference['rotation'] @ np.array([
            reference['depth_m']*section['centering_error_px']/reference['focal_px'][0], 0., 0.])
        correction *= min(1., self.projector.config['max_correction_m'] /
                          max(np.linalg.norm(correction[:2]), 1e-9))
        return dict(original, observed_uv=section['center_uv'],
                    centering_error_px=section['centering_error_px'],
                    correction_base_xy=correction[:2].tolist(),
                    visible_width_px=section['width_px'], near_face_section=section)


def receiver_section(rgb, projected_hull, closing_uv):
    """Associate a complete bright parcel component, then measure its near edge.

    The projected volume only selects a component. Its actual exterior edges,
    away from dark fingers, determine the closing-axis correction.
    """
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = ((hsv[:, :, 2] >= 130)
            & ((hsv[:, :, 1] <= 110) | ((hsv[:, :, 1] <= 180) & (hsv[:, :, 2] >= 190)))
            & ((hsv[:, :, 1] >= 12) | (hsv[:, :, 2] >= 180))
            & ~((hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 100)))
    projected = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(projected, np.asarray(projected_hull, np.int32).reshape(-1, 2), 1)
    projected = cv2.dilate(projected, np.ones((31, 31), np.uint8))
    # A shaded cardboard end can be much darker than its roof. Learn chroma
    # only from the already accepted bright parcel inside the projected volume;
    # extend that material through shade without admitting arbitrary dark blobs.
    # A bright background touching the image boundary is already ineligible
    # as a parcel component. Exclude it before learning material too, rather
    # than letting the projection margin mix its chroma into a parcel seed.
    # Keep that margin: the observed volume is only an approximate silhouette.
    seed_mask = mask & (hsv[:, :, 1] >= 45)
    _, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(seed_mask.astype(np.uint8))
    interior = np.array([x >= 3 and y >= 3 and x+w < width-3 and y+h < height-3
                         for x, y, w, h, area in seed_stats])
    interior[0] = False
    seed = seed_mask & (projected > 0) & interior[seed_labels]
    if np.count_nonzero(seed) >= 80:
        hue = hsv[:, :, 0].astype(float)
        peak = int(np.argmax(np.bincount(hsv[:, :, 0][seed], minlength=180)))
        hue_distance = np.minimum(abs(hue-peak), 180-abs(hue-peak))
        material = seed & (hue_distance <= 5)
        if np.count_nonzero(material) >= .6*np.count_nonzero(seed):
            saturation = float(np.median(hsv[:, :, 1][material]))
            mask |= ((hue_distance <= 5)
                     & (abs(hsv[:, :, 1].astype(float)-saturation) <= 35)
                     & (hsv[:, :, 2] >= 90))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 45, 110)
    # Shaded cardboard and the gray background can have nearly equal luminance
    # while retaining a clear color boundary. Verify that boundary in the RGB
    # channels too; the material mask itself is never treated as an edge.
    for channel in cv2.split(rgb):
        edges |= cv2.Canny(cv2.GaussianBlur(channel, (3, 3), 0), 45, 110)
    lit = cv2.erode((gray >= 45).astype(np.uint8), np.ones((5, 5), np.uint8))
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
    matches = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if (area < 600 or min(w, h) < 20 or w > width*.6 or h > height*.8
                or x < 3 or y < 3 or x+w >= width-3 or y+h >= height-3):
            continue
        if np.count_nonzero((labels == index) & (projected > 0)) < .5*area:
            continue
        sections = []
        # Search nearest the closing line on both sides. A valid silhouette
        # just below that line must not be discarded while the giver obscures
        # the background above it. The same real-edge and finger exclusions
        # apply, with at most 20 pixels of additional downward search.
        rows = range(y+h//3+1, min(y+h-5, int(closing_uv[1])+20)+1)
        for row in sorted(rows, key=lambda row: abs(row-closing_uv[1])):
            columns = np.flatnonzero(labels[row] == index)
            if len(columns) < max(20, .55*w):
                continue
            left, right = int(columns[0]), int(columns[-1])
            if (lit[row, left] and lit[row, right]
                    and distance[row, left] <= 2.8 and distance[row, right] <= 2.8):
                sections.append(((left+right)/2, row, right-left))
                if len(sections) >= 5:
                    break
        if len(sections) < 3:
            continue
        center, row, span = np.median(sections, axis=0)
        matches.append(dict(center_uv=[float(center), float(row)], width_px=float(span),
                            centering_error_px=float(center-closing_uv[0]),
                            edge_rows=len(sections), bbox=[int(x), int(y), int(w), int(h)]))
    return matches[0] if len(matches) == 1 else None
