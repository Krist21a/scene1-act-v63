"""Project a head RGB-D parcel hypothesis into each wrist RGB camera.

The wrist view has no depth stream. Its depth comes from the selected head
candidate and the sensed arm pose; RGB supplies only a bounded local offset.
"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


class WristProjector:
    FRAMES = ('Left Wrist Camera View', 'Right wrist Camera View')

    def __init__(self, rospy, tf_buffer, kinematics, config):
        from sensor_msgs.msg import CameraInfo
        self.kinematics, self.config = kinematics, config
        self.calibration = []
        for side, letter in enumerate(('l', 'r')):
            info = rospy.wait_for_message(f'/cam_{letter}/color/camera_info',
                                          CameraInfo, timeout=5.)
            if info.header.frame_id != self.FRAMES[side]:
                raise ValueError(f'Unexpected {letter} wrist optical frame')
            transform = tf_buffer.lookup_transform(f'zarm_{letter}7_link', self.FRAMES[side],
                                                   rospy.Time(0), rospy.Duration(.5))
            t, q = transform.transform.translation, transform.transform.rotation
            self.calibration.append(dict(
                local_xyz=np.array([t.x, t.y, t.z]),
                local_rotation=Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                intrinsic=np.asarray(info.K, float).reshape(3, 3),
                native_wh=np.array([info.width, info.height], float)))

    def project(self, side, target_base, measured, image_wh, allow_outside=False):
        calibration = self.calibration[side]
        start = (0, 8)[side]
        arm = self.kinematics.matrix(np.asarray(measured)[start:start+7], side)
        rotation = arm[:3, :3] @ calibration['local_rotation']
        camera = (arm[:3, 3] - arm[:3, :3] @ self.kinematics.tool_xyz
                  + arm[:3, :3] @ calibration['local_xyz'])
        point = rotation.T @ (np.asarray(target_base, float) - camera)
        depth = -float(point[2])
        if depth <= .08 or depth >= 2.:
            return None
        scale = np.asarray(image_wh, float) / calibration['native_wh']
        k = calibration['intrinsic']
        # The simulator wrist frames look along -Z; image V runs opposite
        # camera Y. Both signs are checked against captured wrist images.
        uv = np.array([(k[0, 2] + k[0, 0]*point[0]/depth)*scale[0],
                       (k[1, 2] - k[1, 1]*point[1]/depth)*scale[1]])
        if not np.all(np.isfinite(uv)) or (not allow_outside and
                (np.any(uv < 0) or np.any(uv >= image_wh))):
            return None
        return dict(uv=uv, depth_m=depth, rotation=rotation,
                    focal_px=np.array([k[0, 0]*scale[0], k[1, 1]*scale[1]]))

    def measure_edges(self, side, target, measured, rgb):
        """Verify the projected parcel roof against RGB edges, independently of color blobs.

        A pad/parcel merged component cannot establish this correspondence. At
        least two adjacent physical box edges must be visible near the projected
        roof; dark fingers and image boundaries provide no supporting samples.
        """
        height, width = rgb.shape[:2]
        point = np.asarray(target['position_base_m'], float).copy()
        point[2] = target['top_z_base_m']
        major = np.asarray(target.get('major_axis_xy', [1., 0.]), float)
        major /= max(np.linalg.norm(major), 1e-9)
        minor = np.array([-major[1], major[0]])
        extent = np.asarray(target.get('oriented_extent_m', target['extent_xy_m']))
        corners = []
        for a, b in ((-1,-1), (1,-1), (1,1), (-1,1)):
            corner = point.copy()
            corner[:2] += (a*extent[0]*major + b*extent[1]*minor)/2
            projected = self.project(side, corner, measured, [width,height], allow_outside=True)
            if projected is None:
                return None
            corners.append(projected['uv'])
        corners = np.asarray(corners)
        projected = self.project(side, point, measured, [width,height])
        if projected is None or abs(cv2.contourArea(corners.astype(np.float32))) < 300:
            return None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3,3), 0), 45, 110)
        # Finger silhouettes have strong edges too; both sides of the boundary
        # must be sufficiently bright to count as box/table evidence.
        lit = (gray >= 65).astype(np.uint8)
        lit = cv2.erode(lit, np.ones((5,5), np.uint8))
        edges[lit == 0] = 0
        distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
        samples = [corners[i][None,:] + np.linspace(.12,.88,25)[:,None] *
                   (corners[(i+1)%4]-corners[i])[None,:] for i in range(4)]
        best = None
        for dy in range(-20,21,4):
            for dx in range(-20,21,4):
                support = []
                for sample in samples:
                    xy = np.rint(sample + [dx,dy]).astype(int)
                    valid = (xy[:,0] >= 3) & (xy[:,0] < width-3) & (xy[:,1] >= 3) & (xy[:,1] < height-3)
                    hits = np.zeros(len(xy),bool)
                    hits[valid] = distance[xy[valid,1],xy[valid,0]] <= 2.8
                    support.append(float(hits.mean()))
                adjacent = max(min(support[i],support[(i+1)%4]) for i in range(4))
                score = adjacent + .15*np.mean(support) - .001*np.hypot(dx,dy)
                if best is None or score > best[0]:
                    best = (score, dx, dy, support, adjacent)
        _, dx, dy, support, adjacent = best
        if adjacent < self.config.get('min_edge_support', .64):
            return None
        observed = projected['uv'] + [dx,dy]
        center_error = float(observed[0]-width/2)
        local = projected['depth_m'] * np.array([center_error/projected['focal_px'][0],0.,0.])
        correction = projected['rotation'] @ local
        correction *= min(1., self.config['max_correction_m']/max(np.linalg.norm(correction[:2]),1e-9))
        return dict(side=side, projected_uv=projected['uv'].tolist(), observed_uv=observed.tolist(),
                    projection_error_px=float(np.hypot(dx,dy)), centering_error_px=center_error,
                    correction_base_xy=correction[:2].tolist(), depth_m=projected['depth_m'],
                    label='geometry_verified', edge_support=support,
                    roof_polygon_px=(corners+[dx,dy]).tolist())

    def measure_outline(self, side, target, measured, rgb):
        """Center on visible parcel sides without imposing the old roof shape.

        Projection associates the parcel; current exterior edges determine
        alignment. The near cross-section is used because a perspective roof
        centroid can be centered while the closing fingers are still offset.
        """
        height, width = rgb.shape[:2]
        point = np.asarray(target['position_base_m'], float).copy()
        point[2] = target['top_z_base_m']
        projected = self.project(side, point, measured, [width, height])
        if projected is None:
            return None
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Separate the light cardboard face from saturated pad/tape and dark
        # fingers. Color alone is insufficient: exterior edge support follows.
        mask = ((hsv[:,:,2] >= 130)
                & ((hsv[:,:,1] <= 110) | ((hsv[:,:,1] <= 180) & (hsv[:,:,2] >= 190)))
                & ((hsv[:,:,1] >= 12) | (hsv[:,:,2] >= 180))
                & ~((hsv[:,:,0] >= 40) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] >= 100)))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        edges = cv2.Canny(cv2.GaussianBlur(gray, (3,3), 0), 45, 110)
        lit = cv2.erode((gray >= 45).astype(np.uint8), np.ones((5,5), np.uint8))
        edges[lit == 0] = 0
        distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 3)
        arm = self.kinematics.matrix(np.asarray(measured)[(0,8)[side]:(0,8)[side]+7], side)
        tcp = self.project(side, arm[:3,3], measured, [width,height], allow_outside=True)
        closing_row = int(tcp['uv'][1]) if tcp is not None else int(height*.8)
        major = np.asarray(target.get('major_axis_xy',[1.,0.]),float)
        major /= np.linalg.norm(major)
        minor = np.array([-major[1],major[0]])
        extent = target.get('oriented_extent_m',target.get('extent_xy_m'))
        expected_span = np.zeros(2)
        if extent is not None:
            corners = []
            for a,b in ((-1,-1),(1,-1),(1,1),(-1,1)):
                corner = point.copy()
                corner[:2] += (a*extent[0]*major+b*extent[1]*minor)/2
                projected_corner = self.project(side,corner,measured,[width,height],allow_outside=True)
                if projected_corner is None:
                    return None
                corners.append(projected_corner['uv'])
            expected_span = np.ptp(corners,axis=0)
        matches = []
        for component in range(1, count):
            x, y, w, h, area = stats[component]
            if (area < self.config.get('min_area_px',600) or max(w,h) > self.config.get('max_bbox_px',320)
                    or min(w,h) < 20 or x < 3 or y < 3 or x+w >= width-3 or y+h >= height-3):
                continue
            if w < .65*expected_span[0] or h < .65*expected_span[1]:
                continue  # a strip beside the tape is not the whole parcel
            uv = projected['uv']
            if not (x-20 <= uv[0] <= x+w+20 and y-20 <= uv[1] <= y+h+20):
                continue
            contours, _ = cv2.findContours((labels == component).astype(np.uint8),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(max(contours,key=cv2.contourArea))
            polygon = cv2.approxPolyDP(hull,.02*cv2.arcLength(hull,True),True)[:,0,:]
            if not 4 <= len(polygon) <= 8:
                continue
            support = []
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                sample = np.rint(a + np.linspace(.12,.88,25)[:,None]*(b-a)).astype(int)
                support.append(float(np.mean(distance[sample[:,1],sample[:,0]] <= 2.8)))
            if max(min(support[i],support[(i+1)%len(support)]) for i in range(len(support))) < self.config.get('min_edge_support',.64):
                continue
            sections = []
            for row in range(min(y+h-5,closing_row), y+h//3, -1):
                columns = np.flatnonzero(labels[row] == component)
                if len(columns) < max(20,w*.35):
                    continue
                left,right = int(columns[0]),int(columns[-1])
                if (lit[row,left] and lit[row,right] and
                        distance[row,left] <= 2.8 and distance[row,right] <= 2.8):
                    sections.append(((left+right)/2,row,right-left))
                    if len(sections) >= 5:
                        break
            if len(sections) < 3:
                continue
            center,row,span = np.median(sections,axis=0)
            matches.append((center,row,span,polygon,support))
        if len(matches) != 1:
            return None
        center,row,span,polygon,support = matches[0]
        center_error = float(center-width/2)
        local = projected['depth_m'] * np.array([center_error/projected['focal_px'][0],0.,0.])
        correction = projected['rotation'] @ local
        correction *= min(1.,self.config['max_correction_m']/max(np.linalg.norm(correction[:2]),1e-9))
        return dict(side=side,label='outline_verified',projected_uv=projected['uv'].tolist(),
                    observed_uv=[float(center),float(row)],centering_error_px=center_error,
                    correction_base_xy=correction[:2].tolist(),depth_m=projected['depth_m'],
                    edge_support=support,roof_polygon_px=polygon.tolist(),visible_width_px=float(span))

    def aim_error(self, side, target_base, measured, vertical_fraction=.5):
        """Return a base-frame rotation that brings an unseen target into view."""
        calibration = self.calibration[side]
        start = (0, 8)[side]
        arm = self.kinematics.matrix(np.asarray(measured)[start:start+7], side)
        rotation = arm[:3, :3] @ calibration['local_rotation']
        camera = (arm[:3, 3] - arm[:3, :3] @ self.kinematics.tool_xyz
                  + arm[:3, :3] @ calibration['local_xyz'])
        direction = np.asarray(target_base, float) - camera
        distance = np.linalg.norm(direction)
        if distance <= .08 or distance >= 2.:
            return None
        direction /= distance
        # Aim at a usable part of the image. A grasp target need not occupy
        # the principal point, which can require an unreachable wrist angle.
        if vertical_fraction != .5:
            k = calibration['intrinsic']; h = calibration['native_wh'][1]
            ray = np.array([0., -(vertical_fraction*h-k[1,2])/k[1,1], -1.])
            forward = rotation @ (ray/np.linalg.norm(ray))
        else:
            forward = -rotation[:, 2]
        angle = np.arccos(np.clip(np.dot(forward, direction), -1., 1.))
        axis = np.cross(forward, direction)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            return dict(error_base=np.zeros(3), angle_degrees=float(np.rad2deg(angle)))
        return dict(error_base=(axis/axis_norm*angle),
                    angle_degrees=float(np.rad2deg(angle)))

    def measure(self, side, target_base, measured, detections, image_wh,
                max_center_error_px=None):
        projected = self.project(side, target_base, measured, image_wh)
        if projected is None:
            return None
        uv = projected['uv']
        matches = []
        for detection in detections:
            if not detection.get('stable') or detection.get('clipped') or detection.get('candidate_count') != 1:
                continue
            x, y, w, h = detection['bbox']
            if (detection['area_px'] < self.config['min_area_px'] or
                    max(w, h) > self.config['max_bbox_px']):
                continue
            if x-8 <= uv[0] <= x+w+8 and y-8 <= uv[1] <= y+h+8:
                distance = np.linalg.norm(np.asarray(detection['uv'])-uv)
                limit = (self.config['max_center_error_px'] if max_center_error_px is None
                         else max_center_error_px)
                if distance <= limit:
                    matches.append((distance, detection))
        if len(matches) != 1:
            return None
        distance, detection = matches[0]
        # The midpoint between the open fingers is on the image centreline.
        # Use the observed horizontal error to steer that midpoint; the head
        # RGB-D estimate continues to control depth and height.
        center_error = float(detection['uv'][0] - image_wh[0]/2)
        local = projected['depth_m'] * np.array([
            center_error/projected['focal_px'][0], 0., 0.])
        offset = projected['rotation'] @ local
        length = np.linalg.norm(offset[:2])
        if length > self.config['max_correction_m']:
            offset *= self.config['max_correction_m']/length
        return dict(side=side, projected_uv=uv.tolist(), observed_uv=detection['uv'],
                    projection_error_px=float(distance), centering_error_px=center_error,
                    correction_base_xy=offset[:2].tolist(),
                    depth_m=projected['depth_m'], label=detection['label'])
