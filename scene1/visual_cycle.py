"""Visual completion after the weighing regrasp; no ACT calls or scene truth.

The grasp transform comes from ResidualAssist's observed parcel. Both arms use
measured FK and bounded Cartesian feedback. Bin geometry comes from RGB-D.
"""
import copy
import cv2
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .motion import action_vector, JOINTS
from .receiver import receiver_section
from .bin_layout import plan_bin_drop, validate_layout, slot_points


def cloud(snapshot, stride=3):
    d, k, m = snapshot['depth_m'], snapshot['intrinsic'], snapshot['camera_to_base']
    v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
    z = d[v, u].ravel(); u, v = u.ravel(), v.ravel()
    p = np.column_stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z))
    p = p @ m[:3, :3].T + m[:3, 3]
    hsv = cv2.cvtColor(snapshot['rgb'], cv2.COLOR_RGB2HSV)[v, u]
    valid = np.isfinite(p).all(axis=1) & (z > .1) & (z < 2.5)
    return p[valid], hsv[valid]


def bin_rim(snapshot, previous=None):
    """Fit the observed elevated green rim, distinct from the weighing pad."""
    p, hsv = cloud(snapshot)
    plane = snapshot['table_z_base_m']
    selected = ((hsv[:, 0] >= 40) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 80)
                & (p[:, 2] > plane + .12) & (p[:, 2] < plane + .5)
                & (p[:, 0] > .15) & (p[:, 0] < 1.2) & (abs(p[:, 1]) < .8))
    if previous is not None:
        # A nearby hand or parcel can contain green pixels higher than the
        # container. Re-observe the already identified rim in its local region.
        lo = previous['corners'].min(axis=0)-.05
        hi = previous['corners'].max(axis=0)+.05
        selected &= (np.all(p[:, :2] >= lo, axis=1) & np.all(p[:, :2] <= hi, axis=1)
                     & (abs(p[:, 2]-previous['rim_z']) < .035))
    p = p[selected]
    if len(p) < 150:
        return None
    top = float(np.percentile(p[:, 2], 95))
    rim = p[abs(p[:, 2]-top) < .009]
    if len(rim) < 80:
        return None
    # The seed changes placement, not the bin's rectangular shape. Fit its
    # orientation as well as center rather than assuming fixed coordinates.
    rect = cv2.minAreaRect(rim[:, :2].astype(np.float32))
    corners = cv2.boxPoints(rect)
    extent = np.sort(rect[1])
    if not (.22 < extent[0] < .55 and .28 < extent[1] < .65):
        return None
    if not rim_edges_supported(rim, corners):
        return None
    return dict(center=np.mean(corners, axis=0), corners=corners, rim_z=top,
                plane_z=plane, stamp=snapshot['stamp'])


def rim_edges_supported(points, corners):
    """A first rectangle needs actual rim samples along all four sides.

    A disconnected green/depth point can stretch an otherwise plausible
    bounding box. The existing tracker may use two edges only after a shape
    has been established; it cannot establish the missing side's location.
    """
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        tangent = b-a
        length = np.linalg.norm(tangent)
        if length <= 1e-9:
            return False
        tangent /= length
        relative = points[:, :2]-a
        along = relative@tangent
        near = ((abs(relative@np.array([-tangent[1], tangent[0]])) < .009)
                & (along >= 0.) & (along <= length))
        samples = np.sort(along[near])
        groups = np.split(samples, np.flatnonzero(np.diff(samples) > .02)+1)
        if not any(len(group) >= 12 and np.ptp(group) >= .08 for group in groups):
            return False
    return True


def inside_bin(points, rim, margin=.025):
    return all(cv2.pointPolygonTest(rim['corners'], tuple(map(float, p[:2])), True) >= margin
               for p in points)


def largest_surface(points, distance=.008):
    """Keep one connected observed patch, excluding isolated color/depth noise."""
    if not len(points):
        return points
    neighbors = cKDTree(points).query_ball_point(points, distance)
    seen, best = set(), []
    for index in range(len(points)):
        if index in seen:
            continue
        pending, component = [index], []
        seen.add(index)
        while pending:
            index = pending.pop(); component.append(index)
            for neighbor in neighbors[index]:
                if neighbor not in seen:
                    seen.add(neighbor); pending.append(neighbor)
        if len(component) > len(best):
            best = component
    return points[best]


def track_bin_rim(snapshot, previous):
    """Track a rigid observed rectangle from two nonparallel visible edges.

    Re-fitting a smaller bounding rectangle during hand occlusion would make
    the container shrink and move. Keep its observed shape and fit translation.
    """
    points, hsv = cloud(snapshot)
    green = (hsv[:,0] >= 40) & (hsv[:,0] <= 90) & (hsv[:,1] >= 80)
    points = points[green & (abs(points[:,2]-previous['rim_z']) < .012)]
    if len(points) < 60:
        return None
    corners = previous['corners']
    vectors = np.roll(corners,-1,axis=0)-corners
    lengths = np.linalg.norm(vectors,axis=1)
    tangent = vectors/lengths[:,None]
    normals = np.column_stack((-tangent[:,1],tangent[:,0]))
    relative = points[:,:2,None]-corners.T[None,:,:]
    distances = np.einsum('nce,ec->ne',relative,normals)
    along = np.einsum('nce,ec->ne',relative,tangent)
    possible = (along >= -.015) & (along <= lengths[None,:]+.015)
    score = np.where(possible,abs(distances),np.inf)
    edge = score.argmin(axis=1)
    good = score[np.arange(len(points)),edge] < .025
    supported = []
    for index in range(4):
        group = good & (edge == index)
        if np.count_nonzero(group) >= 12 and np.ptp(along[group,index]) >= .08:
            supported.append(index)
    if len(supported) < 2 or np.linalg.matrix_rank(normals[supported],tol=.2) < 2:
        return None
    good &= np.isin(edge,supported)
    indices = np.flatnonzero(good)
    if len(indices) < 60:
        return None
    matrix = normals[supported]
    # Normals point inward. Each band contains both the rim's outer edge and
    # inner wall; fit the robust outer envelope, not the band's midpoint.
    target = np.array([np.percentile(distances[good & (edge == index),index],10)
                       for index in supported])
    delta = np.linalg.lstsq(matrix,target,rcond=None)[0]
    residual = abs(matrix@delta-target)
    if np.linalg.norm(delta) > .025 or max(residual) > .009:
        return None
    result = copy.deepcopy(previous)
    result.update(center=previous['center']+delta,
                  corners=(corners+delta).astype(np.float32),
                  rim_z=float(np.percentile(points[good,2],95)), stamp=snapshot['stamp'],
                  visible_edges=len(supported), supporting_points=len(indices))
    return result


class VisualCycle:
    def __init__(self, config, kinematics, low, high, home):
        self.config, self.k = config, kinematics
        validate_layout(config)
        self.low, self.high = action_vector(low), action_vector(high)
        for side, start in enumerate((0, 8)):
            self.low[start:start+7] = np.maximum(self.low[start:start+7], self.k.limits[side][:, 0])
            self.high[start:start+7] = np.minimum(self.high[start:start+7], self.k.limits[side][:, 1])
        self.home = action_vector(home)
        self.active = self.done = False
        self.completed = 0
        self.rim = None
        self.last_scene_stamp = None
        self.stage = 'idle'

    def observe_scene(self, snapshot):
        if snapshot is None or snapshot['stamp'] == self.last_scene_stamp:
            return
        self.last_scene_stamp = snapshot['stamp']
        rim = (bin_rim(snapshot) if self.rim is None else track_bin_rim(snapshot,self.rim))
        if rim is not None:
            self.rim = rim

    def begin(self, assist, measured, previous_sent):
        if not assist.regrasp_complete or assist.pending is not None or assist.geometry[1] is None:
            raise RuntimeError('Visual completion requires confirmed weighing regrasp geometry')
        if self.rim is None:
            raise RuntimeError('No observed bin rim; refusing blind completion')
        self.geometry = copy.deepcopy(assist.geometry[1])
        self.side = 1
        self.command = action_vector(previous_sent)
        self.command[7], self.command[15] = 0., .98
        self.initial_right = self.k.matrix(measured[8:15], 1)
        self.support_z = assist.weighing_target[2]
        self.active, self.done = True, False
        self.total_steps = self.lost = 0
        self.last_feedback = None
        self.goal = [None, self.initial_right.copy()]
        self.goal[1][2, 3] += self.config.get('lift_m', .16)
        self._stage('lift')

    def _stage(self, name):
        self.stage, self.stage_steps, self.stable = name, 0, 0
        self.last_positions = None
        self.evidence_stamp = None
        self.contact_stable = 0
        self.previous_grip = None
        self.bin_evidence = []
        self.servo_previous = [None, None]
        self.servo_precision = [False, False]
        self.home_previous = None
        self.receiver_center_frames = 0

    def world_box(self, measured):
        start = (0, 8)[self.side]
        tool = self.k.matrix(measured[start:start+7], self.side)
        r = Rotation.from_matrix(tool[:3, :3])
        return (tool[:3, 3]+r.apply(self.geometry['center']),
                r*self.geometry['rotation'], np.asarray(self.geometry['extent']))

    def camera_target(self, measured):
        if self.stage in ('left_clear', 'bin_transport', 'release_left', 'left_retreat', 'verify_bin'):
            return np.r_[self.rim['center'], self.rim['rim_z']-.03]
        if self.stage == 'return_home':
            return None
        return self.world_box(measured)[0]

    def _servo(self, measured, side, goal, posture_target=None, posture_gain=.04, descent_alignment=False,
               coupled_pose_request=False):
        start = (0, 8)[side]
        q = measured[start:start+7]
        current = self.k.matrix(q, side)
        r = Rotation.from_matrix(current[:3, :3])
        xyz = goal[:3, 3]-current[:3, 3]
        angle = (Rotation.from_matrix(goal[:3, :3])*r.inv()).as_rotvec()
        position_error, angle_error = float(np.linalg.norm(xyz)), float(np.linalg.norm(angle))
        if descent_alignment:
            # Use the original regrasp descent ordering for an open-hand
            # pickup too. Preserve the final error above for arrival checks.
            xyz[2] = np.clip(xyz[2], -.012, .012) if np.linalg.norm(xyz[:2]) <= .008 else 0.
            finger_tilt = np.arcsin(np.clip(abs(current[2, 0]), 0., 1.))
            if finger_tilt > np.deg2rad(5.) or angle_error > np.deg2rad(8.):
                xyz[:] = 0.
        if position_error < .08 and angle_error < .35:
            self.servo_precision[side] = True
        precision = self.servo_precision[side]
        previous = self.servo_previous[side]
        self.servo_previous[side] = current.copy()
        if previous is not None:
            # Two control intervals of derivative damping; do not integrate
            # every delayed WBC tracking error as a new position demand.
            damping = 4. if precision else 2.
            xyz -= damping*(current[:3, 3]-previous[:3, 3])
            angle -= damping*(r*Rotation.from_matrix(previous[:3, :3]).inv()).as_rotvec()
        jac = np.empty((6, 7))
        for j in range(7):
            shifted = q.copy(); shifted[j] += 1e-4
            m = self.k.matrix(shifted, side)
            jac[:3, j] = (m[:3, 3]-current[:3, 3])/1e-4
            jac[3:, j] = .12*(Rotation.from_matrix(m[:3, :3])*r.inv()).as_rotvec()/1e-4
        xyz_scale = min(1., .012/max(np.linalg.norm(xyz), 1e-9))
        angle_scale = min(1., .045/max(np.linalg.norm(angle), 1e-9))
        if coupled_pose_request:
            # Preserve the weighted pose-error direction when one request
            # saturates. Both original component limits still apply.
            xyz_scale = angle_scale = min(xyz_scale, angle_scale)
        requested = np.r_[xyz*xyz_scale, .12*angle*angle_scale]
        step = self.config.get('max_joint_step_rad', .012)
        gain = self.config.get('servo_gain', .18)
        if precision:
            step *= .5
            gain /= 3.
        lower = np.maximum(-step, self.low[start:start+7]-self.command[start:start+7])
        upper = np.minimum(step, self.high[start:start+7]-self.command[start:start+7])
        matrix = np.vstack((jac, .018*np.eye(7)))
        preferred = (np.zeros(7) if posture_target is None else
                     np.clip(posture_gain*(np.asarray(posture_target)-self.command[start:start+7]), -.02, .02))
        target = np.r_[gain*requested, .018*preferred]
        free = upper-lower > 1e-10
        delta = (lower+upper)/2
        if np.any(free):
            delta[free] = lsq_linear(matrix[:, free], target-matrix[:, ~free]@delta[~free],
                                    bounds=(lower[free], upper[free]), method='bvls').x
        self.command[start:start+7] += delta
        return position_error, angle_error, current[:3, 3]

    def _arrived(self, measured, *, position_limit=None, angle_limit=None, posture_targets=None, posture_gain=.04,
                 descent_alignment=False, coupled_pose_request=False):
        sides = [side for side, goal in enumerate(self.goal) if goal is not None]
        errors = [self._servo(measured, side, self.goal[side],
                  None if posture_targets is None else posture_targets[side], posture_gain,
                  descent_alignment=descent_alignment and side == 1,
                  coupled_pose_request=coupled_pose_request and side == 1) for side in sides]
        positions = np.concatenate([e[2] for e in errors])
        moving = self.last_positions is None or np.max(abs(positions-self.last_positions)) > .004
        self.last_positions = positions
        if angle_limit is None:
            angle_limit = .25 if self.stage in ('left_clear', 'bin_transport') else .14
        # The bin target is an interior waypoint, not a contact pose. Final
        # release independently checks all observed-volume corners against
        # the fresh rim, including 25 mm edge and vertical clearance.
        if position_limit is None:
            position_limit = .025 if self.stage == 'bin_transport' else .012
        empty = lambda side: self.stage == 'left_retreat' or (self.stage == 'right_retreat' and side == 1)
        ok = all(e[0] < position_limit and (empty(side) or e[1] < angle_limit)
                 for side, e in zip(sides, errors)) and not moving
        self.stable = self.stable+1 if ok else 0
        return self.stable >= 3, [dict(position_m=e[0], angle_rad=e[1]) for e in errors]

    def _head_support(self, snapshot, measured):
        if snapshot is None:
            return None
        center, rotation, extent = self.world_box(measured)
        p, hsv = cloud(snapshot)
        local = rotation.inv().apply(p-center)
        green = (hsv[:, 0] >= 40) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 100)
        sel = (~green & (hsv[:, 2] >= 80)
               & np.all(abs(local) <= extent/2+.018, axis=1))
        group = p[sel]
        if len(group) < 30:
            return None
        height = abs(rotation.as_matrix()[2])@extent
        bottom = float(np.percentile(group[:, 2], 95)-height)
        return dict(points=len(group), bottom_m=bottom)

    def _wrist_support(self, assist, measured):
        if assist.wrist_projector is None or 'left' not in assist.wrist_images:
            return None
        center, rotation, extent = self.world_box(measured)
        rgb = assist.wrist_images['left']; h, w = rgb.shape[:2]
        corners = center+rotation.apply([[a, b, c] for a in (-extent[0]/2, extent[0]/2)
            for b in (-extent[1]/2, extent[1]/2) for c in (-extent[2]/2, extent[2]/2)])
        uv = []
        for point in corners:
            projected = assist.wrist_projector.project(0, point, measured, [w, h], allow_outside=True)
            if projected is None:
                return None
            uv.append(projected['uv'])
        hull = cv2.convexHull(np.asarray(uv, np.float32)).astype(np.int32)
        mask = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(mask, hull, 1)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        colored = ((hsv[:, :, 1] >= 12) & (hsv[:, :, 2] >= 90)
                   & ~((hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 100)))
        count = int(np.count_nonzero(colored & (mask > 0)))
        if count < 60:
            return None
        evidence = dict(colored_pixels=count, projected_hull=hull[:, 0].tolist())
        tool = self.k.matrix(measured[:7], 0)
        closing = assist.wrist_projector.project(0, tool[:3, 3], measured, [w, h], allow_outside=True)
        parcel = assist.wrist_projector.project(0, center, measured, [w, h], allow_outside=True)
        if closing is not None and parcel is not None:
            section = receiver_section(rgb, hull, closing['uv'])
            if section is not None:
                correction = parcel['rotation'] @ np.array([
                    parcel['depth_m']*section['centering_error_px']/parcel['focal_px'][0], 0., 0.])
                evidence.update(section, correction_base_m=correction.tolist())
        return evidence

    def _center_receiver(self, measured, evidence, fresh):
        """Estimate a bounded lateral target from observed edges, not a blob's presence."""
        good = evidence is not None and 'correction_base_m' in evidence
        if fresh:
            self.receiver_center_frames = (self.receiver_center_frames+1 if good
                and abs(evidence['centering_error_px']) <= 10. else 0)
        if not good:
            return False
        tool = self.k.matrix(measured[:7], 0)
        axis = self.goal[0][:3, 0]
        offset = float((tool[:3, 3]+np.asarray(evidence['correction_base_m'])
                        -self.receiver_nominal) @ axis)
        self.receiver_offset = .85*self.receiver_offset+.15*np.clip(offset, -.035, .035)
        self.goal[0][:3, 3] = self.receiver_nominal+axis*self.receiver_offset
        return self.receiver_center_frames >= 3

    def _receiver_goal(self, measured, stand_off):
        center, rotation, extent = self.world_box(measured)
        major, width = rotation.as_matrix()[:, 0], rotation.as_matrix()[:, 1]
        giver = self.k.matrix(measured[8:15], 1)[:3, 3]-center
        if major @ giver > 0:
            major, width = -major, -width
        # Opposite end of the observed long axis, width aligned to closing fingers.
        if width[0] > 0:
            width = -width
        y = np.cross(major, width); y /= np.linalg.norm(y)
        width = np.cross(y, major)
        goal = np.eye(4)
        goal[:3, :3] = np.column_stack((width, y, major))
        goal[:3, 3] = center+major*(extent[0]/2-.012+stand_off)
        return goal

    def _handover_goal(self):
        center = np.array([.28, -.015, .25])
        choices = []
        for yaw in (90, -90):
            box_rotation = Rotation.from_euler('z', yaw, degrees=True)
            rotation = box_rotation*self.geometry['rotation'].inv()
            goal = np.eye(4); goal[:3, :3] = rotation.as_matrix()
            goal[:3, 3] = center-rotation.apply(self.geometry['center'])
            choices.append(goal)
        # Principal axes have no signed direction. Keep the giver on the
        # robot's right, leaving the opposite end available to the left hand.
        goal = min(choices, key=lambda g: g[1, 3])
        if goal[1, 3] >= center[1]-.015:
            raise RuntimeError('Observed grasp does not leave an opposite handover end')
        return goal

    def _bin_points(self, snapshot):
        if snapshot is None:
            return np.empty((0, 3))
        # A parcel behind the near wall may expose only a small patch. Keep
        # full sensor resolution for this stationary confirmation step.
        p, hsv = cloud(snapshot, stride=1)
        green = (hsv[:, 0] >= 40) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 100)
        selected = (~green & (hsv[:, 1] >= 12) & (hsv[:, 2] >= 90)
                    & (p[:, 2] < self.rim['rim_z']-.012) & (p[:, 2] > self.rim['plane_z']-.03))
        selected &= (np.all(p[:, :2] >= self.rim['corners'].min(axis=0), axis=1)
                     & np.all(p[:, :2] <= self.rim['corners'].max(axis=0), axis=1))
        group = p[selected]
        return np.asarray([point for point in group if inside_bin([point], self.rim, .015)]).reshape(-1, 3)

    def update(self, measured, snapshot, assist):
        measured = action_vector(measured)
        # A paused simulator republishes identical values with new timestamps.
        # Detect a frozen plant before integrating more movement into its
        # pending command; timestamps alone cannot establish live feedback.
        previous_feedback = getattr(self, 'feedback_sample', None)
        if previous_feedback is not None and np.array_equal(measured, previous_feedback):
            self.unchanged_feedback_steps += 1
            if (self.unchanged_feedback_steps >= 20
                    and max(abs(self.command[JOINTS]-self.feedback_command[JOINTS])) > .01):
                raise RuntimeError('Robot feedback values are frozen while arm commands change; simulation may be paused')
        else:
            self.unchanged_feedback_steps = 0
            self.feedback_sample = measured.copy()
            self.feedback_command = self.command.copy()
        self.total_steps += 1; self.stage_steps += 1
        timeout = 80 if self.stage == 'verify_bin' else self.config.get('stage_timeout_steps', 320)
        if self.stage_steps > timeout:
            raise RuntimeError(f'Visual cycle timed out at {self.stage}; last grip command retained')
        if snapshot is None:
            self.missing_rgbd = getattr(self, 'missing_rgbd', 0)+1
            self.servo_previous = [None, None]
            self.stable = self.contact_stable = 0
            if self.missing_rgbd >= 10:
                raise RuntimeError('Visual cycle requires fresh registered RGB-D; 10 consecutive reads failed')
            return self.command.copy(), dict(state='visual_cycle_sensor_hold', stage=self.stage,
                missing_rgbd_frames=self.missing_rgbd, strategy='visual_motion', policy_controls_arms=False)
        self.missing_rgbd = 0
        fresh = snapshot['stamp'] != self.evidence_stamp
        if fresh:
            self.evidence_stamp = snapshot['stamp']
        grip = (7, 15)[self.side]
        if self.stage not in ('release_left', 'left_retreat', 'verify_bin', 'return_home'):
            self.lost = self.lost+1 if not .2 < measured[grip] < .85 else 0
            if self.lost >= 4:
                raise RuntimeError(f'Payload gripper contact lost in {self.stage}')
        info = dict(state='visual_cycle_'+self.stage, stage=self.stage, phase_steps=self.stage_steps,
                    completed_parcels=self.completed, strategy='visual_motion', policy_controls_arms=False)
        center, rotation, extent = self.world_box(measured)
        if self.stage == 'left_clear' and self.stage_steps == 1:
            self.goal[0][:3, :3] = self.k.matrix(measured[:7], 0)[:3, :3]
        if self.stage == 'transport':
            self.goal[1] = self._handover_goal()
        attached = self.stage not in ('release_left', 'left_retreat', 'verify_bin', 'return_home')
        info.update(parcel_center_base_m=center.tolist() if attached else None, parcel_extent_m=extent.tolist(),
                    parcel_pose_source='grasp_transform_estimate' if attached else None,
                    goal_tcp=[g[:3, 3].tolist() if g is not None else None for g in self.goal],
                    bin_center_xy_m=self.rim['center'].tolist(), bin_rim_z_m=self.rim['rim_z'])
        receiver_aligned = False
        if self.stage in ('left_approach', 'left_insert'):
            tool = self.k.matrix(measured[:7], 0)
            near = np.linalg.norm(tool[:3, 3]-self.goal[0][:3, 3]) < .04
            angle = (Rotation.from_matrix(self.goal[0][:3, :3])*Rotation.from_matrix(tool[:3, :3]).inv()).magnitude()
            evidence = self._wrist_support(assist, measured) if near and angle < .2 else None
            info['receiver_visual'] = evidence
            receiver_aligned = self._center_receiver(measured, evidence, fresh)
            if self.stage == 'left_insert':
                # Advance only while exterior edges remain centered. Keep each
                # demand close to measured progress so an obstruction cannot
                # accumulate the entire insertion as a forceful command.
                if receiver_aligned:
                    axis = self.goal[0][:3, 2]
                    progress = float((self.receiver_insert_start-tool[:3, 3]) @ axis)
                    self.receiver_insert_m = min(.08, self.receiver_insert_m+.001, max(0., progress)+.006)
                    self.receiver_nominal = self.receiver_insert_start-axis*self.receiver_insert_m
                    self.goal[0][:3, 3] = self.receiver_nominal+self.goal[0][:3, 0]*self.receiver_offset
                info['receiver_insert_m'] = self.receiver_insert_m
            info['receiver_lateral_offset_m'] = self.receiver_offset
        if self.stage in ('lift', 'transport', 'left_approach', 'left_insert', 'release_right', 'right_retreat', 'left_clear', 'bin_transport', 'left_retreat'):
            arrived, errors = self._arrived(measured)
            info['pose_errors'] = errors
        else:
            arrived = False
        if self.stage == 'lift':
            evidence = self._head_support(snapshot, measured)
            info['lift_visual'] = evidence
            if arrived and evidence is not None and evidence['bottom_m'] > self.support_z+.06:
                self.goal[1] = self._handover_goal()
                self._stage('transport')
        elif self.stage == 'transport' and arrived:
            self.goal[0] = self._receiver_goal(measured, .08)
            self.receiver_nominal = self.goal[0][:3, 3].copy()
            self.receiver_offset = 0.
            self._stage('left_approach')
        elif self.stage == 'left_approach' and arrived and receiver_aligned:
            self.receiver_insert_start = self.receiver_nominal.copy()
            self.receiver_insert_m = 0.
            self._stage('left_insert')
        elif self.stage == 'left_insert':
            if arrived and receiver_aligned and self.receiver_insert_m >= .08:
                self.command[7] = .98
                self._stage('close_left')
        elif self.stage == 'close_left':
            stable = self.previous_grip is not None and abs(measured[7]-self.previous_grip) < .015
            if fresh:
                self.contact_stable = self.contact_stable+1 if .2 < measured[7] < .85 and stable else 0
                self.previous_grip = measured[7]
            info['receiver_contact_frames'] = self.contact_stable
            if self.contact_stable >= 3:
                center, rotation, extent = self.world_box(measured)
                left = self.k.matrix(measured[:7], 0); lr = Rotation.from_matrix(left[:3, :3])
                self.receiver_geometry = dict(center=lr.inv().apply(center-left[:3, 3]),
                                              rotation=lr.inv()*rotation, extent=extent.copy())
                self.goal[0] = left.copy()
                self.command[15] = 0.
                self.side, self.geometry = 0, self.receiver_geometry
                self._stage('release_right')
            elif self.stage_steps >= 45:
                raise RuntimeError('Left fingers failed to establish contact; right remains closed')
        elif self.stage == 'release_right':
            if fresh:
                self.contact_stable = self.contact_stable+1 if measured[15] <= .15 else 0
            if self.contact_stable >= 3:
                right = self.k.matrix(measured[8:15], 1)
                self.goal[1] = right.copy()
                self.goal[1][:3, 3] += .12*right[:3, 2]+[0, 0, .04]
                self._stage('right_retreat')
        elif self.stage == 'right_retreat' and arrived:
            left = self.k.matrix(measured[:7], 0)
            self.goal = [left.copy(), None]
            self.goal[0][:3, 3] += [.03, 0., .15]
            self._stage('left_clear')
        elif self.stage == 'left_clear' and arrived:
            if snapshot['stamp']-self.rim['stamp'] > 1.:
                raise RuntimeError('Bin rim not visible after turning toward bin')
            # Turn while extending toward the bin. This orientation is not
            # reachable at the folded, near-chest lift waypoint.
            self.goal[0], self.bin_slot = plan_bin_drop(
                self.rim, self.geometry, self.config, self.completed)
            info['bin_slot'] = self.bin_slot
            self._stage('bin_transport')
        elif self.stage == 'bin_transport' and arrived:
            center, rotation, extent = self.world_box(measured)
            corners = center+rotation.apply([[a, b, c] for a in (-extent[0]/2, extent[0]/2)
                for b in (-extent[1]/2, extent[1]/2) for c in (-extent[2]/2, extent[2]/2)])
            fit = inside_bin(corners, self.rim, .025) and min(corners[:, 2]) > self.rim['rim_z']+.025
            info['bin_footprint_clear'] = bool(fit)
            if fit and snapshot['stamp']-self.rim['stamp'] <= 1.:
                self.bin_before = self._bin_points(snapshot)
                self.command[7] = 0.
                self._stage('release_left')
        elif self.stage == 'release_left':
            if fresh:
                self.contact_stable = self.contact_stable+1 if measured[7] <= .15 else 0
            if self.contact_stable >= 3:
                left = self.k.matrix(measured[:7], 0)
                self.goal[0] = left.copy()
                self.goal[0][:3, 3] += .10*left[:3, 2]+[0., 0., .08]
                self._stage('left_retreat')
        elif self.stage == 'left_retreat' and arrived:
            left = self.k.matrix(measured[:7], 0)
            wrist = left[:3, 3]-left[:3, :3]@np.asarray(getattr(self.k, 'tool_xyz', [0,0,-.1488]))
            clear = min(left[2, 3], wrist[2]) > self.rim['rim_z']+.08
            info['empty_hand_above_rim'] = bool(clear)
            if clear:
                self._stage('verify_bin')
        elif self.stage == 'verify_bin' and fresh:
            points = self._bin_points(snapshot)
            info['bin_points_before_slot_association'] = len(points)
            points = slot_points(points, self.rim, self.config, self.completed)
            info['bin_verification_slot'] = self.completed
            if len(self.bin_before) and len(points):
                points = points[cKDTree(self.bin_before).query(points)[0] > .015]
            info['new_bin_points_before_clustering'] = len(points)
            points = largest_surface(points)
            info['new_bin_points'] = len(points)
            if len(points) >= 80 and np.count_nonzero(np.ptp(points, axis=0) >= .01) >= 2:
                center = np.median(points, axis=0)
                self.bin_evidence.append((snapshot['stamp'], center))
                self.bin_evidence = self.bin_evidence[-15:]
                span = np.ptp([p[1] for p in self.bin_evidence], axis=0)
                stable = len(self.bin_evidence) >= 5 and max(span) <= .015
                if stable and self.bin_evidence[-1][0]-self.bin_evidence[0][0] >= 1.:
                    info['bin_visual_confirmed'] = True
                    self.completed += 1
                    self._stage('return_home')
            else:
                self.bin_evidence = []
        elif self.stage == 'return_home':
            delta = self.home[JOINTS]-measured[JOINTS]
            previous = getattr(self, 'home_previous', None)
            velocity = np.zeros(len(JOINTS)) if previous is None else measured[JOINTS]-previous
            self.home_previous = measured[JOINTS].copy()
            change = .06*delta-.24*velocity
            scale = min(1., .012/max(max(abs(change)), 1e-9))
            self.command[JOINTS] += scale*change
            self.command = np.clip(self.command, self.low, self.high)
            self.command[[7, 15]] = 0.
            ready = max(abs(delta)) < .06
            self.stable = self.stable+1 if ready else 0
            if self.stable >= 5:
                self.done, self.active = True, False
                info['state'] = 'visual_cycle_complete'
                info['completed_parcels'] = self.completed
        if info['stage'] != self.stage:
            info['transition_to'] = self.stage
        return np.clip(self.command.copy(), self.low, self.high), info

    def state_dict(self):
        def encode(value):
            if isinstance(value, Rotation):
                return {'rotation_xyzw': value.as_quat().tolist()}
            if isinstance(value, np.ndarray):
                return {'array': value.tolist(), 'dtype': str(value.dtype)}
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            return value
        return {k: encode(v) for k, v in vars(self).items() if k not in ('k', 'config', 'low', 'high')}

    def restore(self, saved, measured, expected_measured):
        def decode(value):
            if isinstance(value, dict):
                if set(value) == {'rotation_xyzw'}:
                    return Rotation.from_quat(value['rotation_xyzw'])
                if set(value) == {'array', 'dtype'}:
                    return np.asarray(value['array'], dtype=value['dtype'])
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value
        measured, expected = action_vector(measured), action_vector(expected_measured)
        restored = decode(saved)
        sent = action_vector(restored['command'])
        # The last issued trajectory can finish after logging stops. Accept
        # settling toward that exact command, but no unrelated joint excursion.
        lower = np.minimum(expected[JOINTS], sent[JOINTS])-.15
        upper = np.maximum(expected[JOINTS], sent[JOINTS])+.15
        if np.any(measured[JOINTS] < lower) or np.any(measured[JOINTS] > upper):
            raise RuntimeError('Visual resume differs from the recorded physical arm state')
        if any(key in restored for key in ('k', 'config', 'low', 'high')):
            raise ValueError('Checkpoint cannot replace controller geometry/limits/configuration')
        if not restored.get('active') or restored.get('done'):
            raise RuntimeError('Visual resume requires an interrupted active cycle')
        if restored['stage'] in ('left_approach', 'left_insert') and 'receiver_nominal' not in restored:
            raise RuntimeError('Legacy receiver checkpoint lacks visual centering state; use a fresh scene')
        if restored['stage'] not in ('release_left','left_retreat','verify_bin','return_home'):
            if not .2 < measured[(7,15)[restored['side']]] < .85:
                raise RuntimeError('Visual resume no longer has payload contact')
        self.__dict__.update(restored)
        self.command = action_vector(self.command)
        self.stage_steps, self.stable = 0, 0
        self.servo_previous = [None, None]
        self.servo_precision = [False, False]
        self.home_previous = None
        self.feedback_sample = None
