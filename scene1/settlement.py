"""Visual evidence that an opened parcel has settled on its observed support."""
from collections import deque
import numpy as np
import cv2


def track_resting_roof(snapshot, target, center_hint):
    """Track the known support-height roof without merging a nearby gripper.

    An absent or substantially occluded roof returns unknown, never zero motion.
    """
    if snapshot is None or 'depth_m' not in snapshot:
        return None
    major = np.asarray(target.get('major_axis_xy',[1.,0.]),float)
    major /= np.linalg.norm(major)
    axes = np.column_stack((major,[-major[1],major[0]]))
    size = np.asarray(target.get('oriented_extent_m',target['extent_xy_m']))
    top = target['top_z_base_m'] + snapshot['table_z_base_m']-target['support_z_base_m']
    depth = snapshot['depth_m']; k = snapshot['intrinsic']
    v,u = np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
    v,u = v.ravel(),u.ravel(); z=depth[v,u]
    camera = np.column_stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z))
    matrix = snapshot['camera_to_base']
    points = camera@matrix[:3,:3].T+matrix[:3,3]
    local = (points[:,:2]-center_hint)@axes
    hsv = cv2.cvtColor(snapshot['rgb'],cv2.COLOR_RGB2HSV)[v,u]
    selected = ((z > .1) & np.isfinite(points).all(axis=1)
                & np.all(abs(local) <= size/2+.03,axis=1)
                & (abs(points[:,2]-top) <= .010) & (hsv[:,2] >= 90)
                & ~((hsv[:,0]>=40)&(hsv[:,0]<=85)&(hsv[:,1]>=100)))
    group = points[selected]
    if len(group) < 45:
        return None
    xy = group[:,:2]@axes
    low,high = np.percentile(xy,[5,95],axis=0)
    observed_size = high-low
    if np.any(observed_size < .8*size) or np.any(observed_size > 1.25*size):
        return None
    return dict(center_xy_m=((low+high)/2@axes.T).tolist(),
                extent_xy_m=observed_size.tolist(), points=len(group))


def resting_observation(snapshot, surface, extent, roof_normal):
    if snapshot is None or 'depth_m' not in snapshot:
        return None
    pad = np.asarray(surface['position_base_m'])
    matches = []
    for candidate in snapshot['candidates']:
        size = np.asarray(candidate.get('oriented_extent_m', candidate['extent_xy_m']))
        ratio = size / np.asarray(extent)[:2]
        if (np.linalg.norm(np.asarray(candidate['position_base_m'])[:2]-pad[:2]) < .08
                and .8 <= ratio[0] <= 1.25 and .75 <= ratio[1] <= 1.35):
            matches.append(candidate)
    if len(matches) != 1:
        return None
    candidate = matches[0]
    center = np.asarray(candidate['position_base_m'])
    major = np.asarray(candidate.get('major_axis_xy', [1.,0.]))
    minor = np.array([-major[1],major[0]])
    size = np.asarray(candidate.get('oriented_extent_m',candidate['extent_xy_m']))
    footprint = surface.get('footprint_xy_m')
    if footprint is None:
        return None
    corners = [center[:2]+a*size[0]/2*major+b*size[1]/2*minor
               for a,b in ((-1,-1),(1,-1),(1,1),(-1,1))]
    margin = min(cv2.pointPolygonTest(np.asarray(footprint,np.float32),tuple(p),True) for p in corners)
    depth = snapshot['depth_m']; k = snapshot['intrinsic']
    v,u = np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
    v,u = v.ravel(),u.ravel(); z = depth[v,u]
    camera = np.column_stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z))
    matrix = snapshot['camera_to_base']
    points = camera@matrix[:3,:3].T+matrix[:3,3]
    delta = points[:,:2]-center[:2]
    hsv = cv2.cvtColor(snapshot['rgb'],cv2.COLOR_RGB2HSV)[v,u]
    selected = ((z > .1) & np.isfinite(points).all(axis=1)
                & (abs(delta@major) <= size[0]/2+.008)
                & (abs(delta@minor) <= size[1]/2+.008)
                & (points[:,2] > snapshot['table_z_base_m']+.025)
                & (points[:,2] < candidate['top_z_base_m']+.015)
                & (hsv[:,2] >= 65))
    normal = roof_normal(points[selected]) if np.count_nonzero(selected) >= 30 else None
    if normal is None:
        return None
    tilt = float(np.rad2deg(np.arccos(np.clip(normal[2],-1,1))))
    height = float(candidate['top_z_base_m']-snapshot['table_z_base_m'])
    # Top-face height alone cannot establish contact; compare it with the
    # previously observed parcel height and the allowed residual tilt.
    expected_height = float(extent[2])
    support_height_ok = abs(height-expected_height) <= .018
    return dict(target=candidate, center_xy_m=center[:2].tolist(), top_height_m=height,
                tilt_degrees=tilt, footprint_margin_m=margin,
                support_height_ok=bool(support_height_ok), stamp=float(snapshot['stamp']))


class SettlementMonitor:
    def __init__(self, duration_s=1., max_xy_span_m=.006, max_height_span_m=.006):
        self.duration_s = duration_s
        self.max_xy_span_m = max_xy_span_m
        self.max_height_span_m = max_height_span_m
        self.samples = deque(maxlen=60)

    def update(self, observation):
        if observation is None:
            self.samples.clear()
            return dict(confirmed=False, reason='missing_geometry')
        stamp = observation['stamp']
        if self.samples and stamp <= self.samples[-1]['stamp']:
            return dict(confirmed=False, reason='duplicate_or_old_frame')
        if self.samples and stamp-self.samples[-1]['stamp'] > .4:
            self.samples.clear()
        valid = (observation['support_height_ok'] and observation['footprint_margin_m'] >= .01
                 and observation['tilt_degrees'] <= 15.)
        if not valid:
            self.samples.clear()
            return dict(confirmed=False, reason='unsupported_or_outside_pad', **observation)
        self.samples.append(observation)
        while len(self.samples) > 2 and stamp-self.samples[1]['stamp'] >= self.duration_s:
            self.samples.popleft()
        xy = np.array([s['center_xy_m'] for s in self.samples])
        span = float(np.max(np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)))
        height_span = float(np.ptp([s['top_height_m'] for s in self.samples]))
        tilt_span = float(np.ptp([s['tilt_degrees'] for s in self.samples]))
        duration = stamp-self.samples[0]['stamp']
        confirmed = (len(self.samples) >= 5 and duration >= self.duration_s
                     and span <= self.max_xy_span_m and height_span <= self.max_height_span_m
                     and tilt_span <= 4.)
        return dict(confirmed=bool(confirmed), reason='stable' if confirmed else 'observing',
                    duration_s=duration, xy_span_m=span, height_span_m=height_span,
                    tilt_span_degrees=tilt_span, **observation)
