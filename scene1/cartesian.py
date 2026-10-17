"""Freeze ACT briefly before a pickup, align XY by RGB-D + IK, then resume."""
import copy
import numpy as np
from .motion import JOINTS, action_vector


class ArmKinematics:
    def __init__(self, rospy):
        from kuavo_humanoid_sdk.msg.kuavo_msgs.srv import fkSrv, twoArmHandPoseCmdSrv
        self.rospy = rospy
        rospy.wait_for_service("/ik/fk_srv", timeout=3)
        rospy.wait_for_service("/ik/two_arm_hand_pose_cmd_srv", timeout=3)
        self.fk_service = rospy.ServiceProxy("/ik/fk_srv", fkSrv)
        self.ik_service = rospy.ServiceProxy("/ik/two_arm_hand_pose_cmd_srv", twoArmHandPoseCmdSrv)

    def fk(self, action):
        result = self.fk_service(action_vector(action)[JOINTS].tolist())
        if not result.success:
            raise RuntimeError("FK service failed")
        return result.hand_poses

    def ik(self, poses, seed, orientation_tolerance=.10):
        from kuavo_humanoid_sdk.msg.kuavo_msgs.msg import twoArmHandPoseCmd
        request = twoArmHandPoseCmd()
        request.frame = 2  # local base frame, same welded base_link as FK
        request.use_custom_ik_param = True
        request.ik_param.major_optimality_tol = 1e-4
        request.ik_param.major_feasibility_tol = 1e-5
        request.ik_param.minor_feasibility_tol = 1e-5
        request.ik_param.major_iterations_limit = 100
        request.ik_param.oritation_constraint_tol = 0.08
        request.ik_param.pos_constraint_tol = 0.005
        request.ik_param.pos_cost_weight = 0.0
        request.ik_param.constraint_mode = 3  # hard position and orientation
        request.hand_poses = poses
        request.joint_angles_as_q0 = True
        request.hand_poses.left_pose.joint_angles = seed[:7].tolist()
        request.hand_poses.right_pose.joint_angles = seed[8:15].tolist()
        result = self.ik_service(request)
        if not result.success:
            raise RuntimeError(f"Grasp alignment IK failed: {result.error_reason}")
        joints = np.asarray(result.q_arm, dtype=float)
        if joints.shape != (14,) or not np.isfinite(joints).all():
            raise RuntimeError("IK did not return 14 finite joints")
        return joints


def orientation_error(a, b, axis=None):
    if axis is not None:
        from scipy.spatial.transform import Rotation
        axis = np.asarray(axis) / np.linalg.norm(axis)
        first = Rotation.from_quat(a).apply(axis)
        second = Rotation.from_quat(b).apply(axis)
        return float(np.arccos(np.clip(first @ second, -1, 1)))
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(2 * np.arccos(np.clip(abs(a @ b), 0, 1)))


class CartesianGraspAligner:
    """Measured lift/translate/descend, grasp dwell and release settlement.

    ACT is frozen throughout a pending manipulation. No joint error integration
    is used: stalled/contact-constrained arms stop rather than accumulating force.
    """
    def __init__(self, config, kinematics, low=None, high=None):
        self.config, self.kinematics = config, kinematics
        self.low = action_vector(low) if low is not None else np.full(16, -np.inf)
        self.high = action_vector(high) if high is not None else np.full(16, np.inf)
        if hasattr(kinematics, 'limits'):
            for side, start in enumerate((0, 8)):
                self.low[start:start + 7] = np.maximum(self.low[start:start + 7], kinematics.limits[side][:, 0])
                self.high[start:start + 7] = np.minimum(self.high[start:start + 7], kinematics.limits[side][:, 1])
        self.previous_grip = [0.0, 0.0]
        self.ready_to_close = [True, True]
        self.closed = [False, False]
        self.offset = np.zeros(16)
        self.grasp_z = [None, None]
        self.initial_grasps = 0
        self.pending = None
        self.held_action = None
        self.target_tracks = []
        self.tick = 0
        self.tracking_bias = np.zeros(16)
        self.loaded_feedforward = np.zeros(16)
        self.held_center_in_tool = [None, None]
        self.held_place_orientation = [None, None]
        self.held_major_axis_in_tool = [None, None]
        self.grasp_tool_height = [None, None]
        self.held_extent = [None, None]
        self.weighing_target = None
        self.last_sent = None
        self.latest_snapshot = None
        self.manipulation_kind = None

    def observe_command(self, sent):
        self.last_sent = action_vector(sent)

    def _remember_targets(self, snapshot):
        self.tick += 1
        self.target_tracks = [t for t in self.target_tracks
                              if self.tick - t['last_seen'] <= self.config['target_memory_steps']]
        if snapshot is None:
            return
        used = set()
        for item in snapshot['candidates']:
            extent = item.get('extent_xy_m')
            if extent is not None and item.get('minor_extent_m', min(extent)) > self.config['max_object_width_m']:
                continue
            position = np.asarray(item['position_base_m'])
            possible = [(float(np.linalg.norm(position[:2] - t['position'][:2])), i)
                        for i, t in enumerate(self.target_tracks) if i not in used]
            distance, index = min(possible, default=(float('inf'), -1))
            if distance <= self.config['target_association_distance_m']:
                track = self.target_tracks[index]
                track['position'] = 0.7 * track['position'] + 0.3 * position
                track['seen'] += 1
                track['last_seen'] = self.tick
                track['item'] = item
                used.add(index)
            else:
                self.target_tracks.append(dict(position=position, seen=1, last_seen=self.tick, item=item))
                used.add(len(self.target_tracks) - 1)

    def _targets(self):
        return [{**t['item'], 'position_base_m': t['position'].tolist(),
                 'occluded_steps': self.tick - t['last_seen']}
                for t in self.target_tracks if t['seen'] >= self.config['target_stable_frames']]

    @staticmethod
    def pose(poses, side):
        return poses.left_pose if side == 0 else poses.right_pose

    def _hold(self, action):
        result = action.copy()
        for side, grip in enumerate((7, 15)):
            if self.closed[side]:
                result[grip] = self.config['closed_command']
            elif self.ready_to_close[side]:
                result[grip] = 0.0
        return np.clip(result, self.low, self.high)

    def _solve(self, poses, side, goal, seed, orientation_tolerance=.10, upright_axis=None):
        poses = copy.deepcopy(poses)
        self.pose(poses, side).pos_xyz = np.asarray(goal).tolist()
        kwargs = dict(orientation_tolerance=orientation_tolerance)
        if hasattr(self.kinematics, 'limits'):
            kwargs['joint_offsets'] = self.loaded_feedforward[JOINTS]
            if upright_axis is not None:
                axes = [None, None]
                axes[side] = upright_axis
                kwargs['upright_axes'] = axes
        joints = self.kinematics.ik(poses, seed, **kwargs)
        result = seed.copy()
        start = 0 if side == 0 else 8
        result[start:start + 7] = joints[side * 7:side * 7 + 7]
        if np.any(result[JOINTS] < self.low[JOINTS]) or np.any(result[JOINTS] > self.high[JOINTS]):
            raise RuntimeError('Grasp IK is outside deployment joint limits; not closing')
        if np.max(np.abs(result[start:start + 7] - seed[start:start + 7])) > self.config['max_joint_residual_rad']:
            raise RuntimeError('Grasp waypoint requires a large joint change; not closing')
        actual = self.pose(self.kinematics.fk(result), side)
        if (np.linalg.norm(np.asarray(actual.pos_xyz) - goal) > 0.008 or
                orientation_error(actual.quat_xyzw, self.pose(poses, side).quat_xyzw, upright_axis) > orientation_tolerance):
            raise RuntimeError('IK/FK round trip does not match the waypoint; not closing')
        return result

    def _grasp_point(self, item, pose):
        from scipy.spatial.transform import Rotation
        point = np.asarray(item['position_base_m'], dtype=float).copy()
        # The 2F85 tool extends along wrist -Z. Horizontal side grasps insert
        # near the front edge; commanding the tool deep into the box collides
        # with its base before the finger pads can straddle the parcel.
        forward = -Rotation.from_quat(pose.quat_xyzw).as_matrix()[:2, 2]
        extent = item.get('extent_xy_m')
        norm = np.linalg.norm(forward)
        if extent is not None and norm > 0.65:
            forward /= norm
            half_length = float(np.abs(forward) @ np.asarray(extent) / 2)
            if item.get('major_axis_xy') is not None and item.get('oriented_extent_m') is not None:
                major = np.asarray(item['major_axis_xy'])
                major /= np.linalg.norm(major)
                minor = np.array([-major[1], major[0]])
                lengths = np.asarray(item['oriented_extent_m'])
                half_length = float((abs(forward @ major) * lengths[0] + abs(forward @ minor) * lengths[1]) / 2)
            point[:2] -= forward * max(0.0, half_length - self.config['side_grasp_insertion_m'])
        return point

    def _begin(self, raw, side, phases, target=None):
        self.held_action = raw.copy()
        self.pending = dict(side=side, phases=phases, index=0, steps=0,
                            stable=0, last_position=None, target=target)

    def _path(self, name, reference, side, goal, seed, grip, gripper_command=0.0):
        from scipy.spatial.transform import Rotation, Slerp
        current = self.pose(self.kinematics.fk(seed), side)
        desired = self.pose(reference, side)
        origin = np.asarray(current.pos_xyz)
        angle = orientation_error(current.quat_xyzw, desired.quat_xyzw)
        count = max(1, int(np.ceil(np.linalg.norm(goal - origin) / 0.025)),
                    int(np.ceil(angle / 0.12)))
        rotations = Slerp([0, 1], Rotation.from_quat([current.quat_xyzw, desired.quat_xyzw]))
        phases = []
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            pose = copy.deepcopy(reference)
            self.pose(pose, side).quat_xyzw = rotations(fraction).as_quat().tolist()
            seed = self._solve(pose, side, origin + fraction * (goal - origin), seed)
            seed[grip] = gripper_command
            phases.append(dict(name=name, action=seed.copy()))
        return phases, seed

    def _parcel_fits(self, center, rotation, surface):
        import cv2
        from scipy.spatial.transform import Rotation
        if self.held_extent[1] is None:
            return True
        extent = np.asarray(self.held_extent[1])
        corners = np.array([[x, y, 0] for x in (-extent[0] / 2, extent[0] / 2)
                             for y in (-extent[1] / 2, extent[1] / 2)])
        if self.held_place_orientation[1] is not None:
            corners = rotation.apply(Rotation.from_quat(self.held_place_orientation[1]).inv().apply(corners))
        corners = corners[:, :2] + np.asarray(center)[:2]
        footprint = surface.get('footprint_xy_m')
        if footprint is None:
            return True
        polygon = np.asarray(footprint, np.float32)
        return all(cv2.pointPolygonTest(polygon, tuple(point.astype(float)), True) >= self.config.get('placement_margin_m', .025) for point in corners)

    def _placement_yaws(self, rotation, surface):
        import cv2
        major = self.held_major_axis_in_tool[1]
        if major is None:
            return (0., -np.pi / 4, np.pi / 4, -np.pi / 2, np.pi / 2)
        heading = rotation.apply(major)
        object_yaw = np.arctan2(heading[1], heading[0])
        object_yaw = (object_yaw + np.pi / 2) % np.pi - np.pi / 2
        pad_yaw = 0.
        if surface.get('footprint_xy_m') is not None:
            rectangle = cv2.minAreaRect(np.asarray(surface['footprint_xy_m'], np.float32))
            pad_yaw = np.deg2rad(rectangle[2])
            pad_yaw = (pad_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        aligned = pad_yaw - object_yaw
        return (aligned, aligned - np.pi / 2, aligned + np.pi / 2, aligned + np.pi)

    def _weighing_path(self, planned, snapshot, reference=None):
        from scipy.spatial.transform import Rotation
        goal = np.asarray(self.kinematics.fk(planned).right_pose.pos_xyz)
        surfaces = snapshot.get('surfaces', [])
        if not surfaces:
            raise RuntimeError('Weighing pad is not visible; keeping parcel grasped')
        surface = min(surfaces, key=lambda s: np.linalg.norm(np.asarray(s['position_base_m'])[:2] - goal[:2]))
        center = np.asarray(surface['position_base_m'])
        direction = -center[:2] / np.linalg.norm(center[:2])
        offsets = [np.zeros(2)]
        if surface.get('footprint_xy_m') is not None:
            offsets += [direction * distance for distance in (.015, .030, .045)]
            offsets += [np.array([x, y]) for x, y in ((-.02, .04), (0, .04), (-.02, .02), (.02, .04))]
        rotation = Rotation.from_quat(self.held_place_orientation[1]) if self.held_place_orientation[1] is not None else Rotation.identity()
        errors = []
        # Turning about the parcel's vertical axis keeps it upright while
        # giving the limited wrist a different, physically feasible branch.
        for yaw in self._placement_yaws(rotation, surface):
            turned = Rotation.from_euler('z', yaw) * rotation
            tested_offsets = offsets
            if surface.get('footprint_xy_m') is not None and self.held_extent[1] is not None:
                polygon = np.asarray(surface['footprint_xy_m'])
                feasible = [np.array([x, y, center[2]])
                            for x in np.arange(polygon[:, 0].min(), polygon[:, 0].max(), .012)
                            for y in np.arange(polygon[:, 1].min(), polygon[:, 1].max(), .012)
                            if self._parcel_fits([x, y, center[2]], turned, surface)]
                feasible.sort(key=lambda point: np.linalg.norm(point[:2]))
                selected = []
                for point in feasible:
                    if all(np.linalg.norm(point[:2] - old[:2]) >= .02 for old in selected):
                        selected.append(point)
                    if len(selected) == 4:
                        break
                tested_offsets = [point[:2] - center[:2] for point in selected] + offsets
            for offset in tested_offsets:
                candidate = center + np.r_[offset, 0]
                if not self._parcel_fits(candidate, turned, surface):
                    continue
                try:
                    return self._weighing_path_at(planned, snapshot, reference, surface, candidate, yaw)
                except RuntimeError as exc:
                    errors.append(str(exc))
        raise RuntimeError('No reachable placement with parcel footprint inside pad: ' + (errors[-1] if errors else 'insufficient visible pad area'))

    def _weighing_path_at(self, planned, snapshot, reference, surface, pad_center, yaw=0.):
        from scipy.spatial.transform import Rotation
        current_poses = self.kinematics.fk(planned)
        poses = current_poses if reference is None else reference
        poses = copy.deepcopy(poses)
        poses.left_pose = copy.deepcopy(current_poses.left_pose)
        if self.held_place_orientation[1] is not None:
            poses.right_pose.quat_xyzw = (Rotation.from_euler('z', yaw) * Rotation.from_quat(self.held_place_orientation[1])).as_quat().tolist()
        pose = poses.right_pose
        goal = np.asarray(current_poses.right_pose.pos_xyz)
        center = pad_center.copy()
        self.weighing_target = center.copy()
        if np.linalg.norm(center[:2] - goal[:2]) > .18:
            raise RuntimeError('Visible weighing pad is too far from ACT release; keeping parcel grasped')
        relative = self.held_center_in_tool[1]
        if relative is not None:
            center = center - Rotation.from_quat(pose.quat_xyzw).apply(relative)
        # Lift vertically before moving a parcel that may already touch the pad.
        goal[2] = max(goal[2] + .08, snapshot['table_z_base_m'] + .22)
        phases, planned = self._path('lift_before_placement', current_poses, 1, goal, planned, 15,
                                    self.config['closed_command'])
        goal[:2] = center[:2]
        segment, planned = self._path('center_above_weighing', poses, 1, goal, planned, 15,
                                     self.config['closed_command'])
        phases.extend(segment)
        height = self.config['weighing_release_height_m']
        if self.grasp_tool_height[1] is not None:
            height = max(height, self.grasp_tool_height[1] + .004)
        release_z = snapshot['table_z_base_m'] + height
        count = max(1, int(np.ceil((goal[2] - release_z) / .015)))
        upright_axis = None
        upright_tolerance = self.config['orientation_tolerance_rad']
        if self.held_place_orientation[1] is not None and self.held_major_axis_in_tool[1] is None:
            upright_axis = Rotation.from_quat(self.held_place_orientation[1]).inv().apply([0, 0, 1])
        for z in np.linspace(goal[2], release_z, count + 1)[1:]:
            goal[2] = z
            try:
                planned = self._solve(poses, 1, goal, planned,
                                      orientation_tolerance=upright_tolerance if self.held_place_orientation[1] is not None else .20,
                                      upright_axis=upright_axis)
                if relative is not None and upright_axis is not None:
                    # A yaw change alters the grasp offset. Center the parcel,
                    # not just the pinch point, after choosing a feasible yaw.
                    actual = self.kinematics.fk(planned).right_pose
                    goal[:2] = (self.weighing_target - Rotation.from_quat(actual.quat_xyzw).apply(relative))[:2]
                    planned = self._solve(poses, 1, goal, planned, orientation_tolerance=upright_tolerance,
                                          upright_axis=upright_axis)
            except RuntimeError as exc:
                previous_z = self.kinematics.fk(planned).right_pose.pos_xyz[2]
                if previous_z - snapshot['table_z_base_m'] <= self.config['max_weighing_release_height_m']:
                    break
                raise RuntimeError(f'No feasible upright release below {previous_z - snapshot["table_z_base_m"]:.3f} m above pad: {exc}') from exc
            phases.append(dict(name='lower_before_release', action=planned.copy()))
        final_pose = self.kinematics.fk(planned).right_pose
        final_rotation = Rotation.from_quat(final_pose.quat_xyzw)
        final_center = np.asarray(final_pose.pos_xyz)
        if relative is not None:
            final_center = final_center + final_rotation.apply(relative)
        if not self._parcel_fits(final_center, final_rotation, surface):
            raise RuntimeError('Reachable release would leave parcel footprint outside pad')
        return phases, planned

    @staticmethod
    def restore_surface_view(snapshot, records):
        """Associate visible pad fragments with an earlier observed boundary."""
        import cv2
        result = dict(snapshot)
        historical = [surface for row in records for surface in row.get('rgbd_surfaces', [])
                      if not surface.get('partial') and surface.get('footprint_xy_m')]
        historical.sort(key=lambda s: np.prod(s['extent_xy_m']), reverse=True)
        surfaces = []
        for observed in snapshot.get('surfaces', []):
            if not observed.get('partial'):
                surfaces.append(observed)
                continue
            points = observed.get('footprint_xy_m', [])
            for old in historical:
                polygon = np.asarray(old['footprint_xy_m'], np.float32)
                if (abs(old['position_base_m'][2] - observed['position_base_m'][2]) < .012
                        and points and all(cv2.pointPolygonTest(polygon, tuple(p), True) > -.008 for p in points)):
                    surface = dict(old)
                    surface['position_base_m'] = [*old['position_base_m'][:2], snapshot['table_z_base_m']]
                    surface['boundary_from_visual_history'] = True
                    surface['visible_fragment'] = observed
                    surfaces.append(surface)
                    break
        result['surfaces'] = surfaces
        return result

    def restore_weighing(self, records, measured, snapshot):
        from scipy.spatial.transform import Rotation
        grasps = [r for r in records if r['cartesian_alignment'].get('state') == 'close_dwell'
                  and r['cartesian_alignment'].get('arm') == 1
                  and r['cartesian_alignment'].get('target') is not None]
        if not grasps or records[-1]['sent_action'][15] < .75:
            raise RuntimeError('Resume log does not describe a held right-hand parcel')
        last = grasps[-1]
        target = last['cartesian_alignment']['target']
        self.held_extent[1] = target.get('extent_xy_m')
        pose = self.kinematics.fk(last['measured_state']).right_pose
        if target.get('major_axis_xy') is not None:
            self.held_major_axis_in_tool[1] = Rotation.from_quat(pose.quat_xyzw).inv().apply([*target['major_axis_xy'], 0])
        self.held_place_orientation[1] = np.asarray(pose.quat_xyzw)
        plane = last.get('rgbd_table_z_base_m')
        if plane is None and last.get('rgbd_surfaces'):
            plane = float(np.median([s['position_base_m'][2] for s in last['rgbd_surfaces']]))
        if plane is not None:
            self.grasp_tool_height[1] = float(pose.pos_xyz[2] - plane)
        self.held_center_in_tool[1] = Rotation.from_quat(pose.quat_xyzw).inv().apply(
            np.asarray(target['position_base_m']) - pose.pos_xyz)
        if np.linalg.norm(self.held_center_in_tool[1]) > .12:
            raise RuntimeError('Restored parcel/tool offset exceeds expected size')
        self.closed[1] = True
        self.ready_to_close[1] = False
        planned = np.clip(action_vector(measured), self.low, self.high)
        planned[[7, 15]] = [0, self.config['closed_command']]
        # Recover the static command/feedback offset while the previous command
        # is still held. Keep it fixed; do not integrate a contact-stalled error.
        offset = action_vector(records[-1]['sent_action'])[8:15] - planned[8:15]
        cap = self.config.get('max_loaded_feedforward_rad', .60)
        if np.max(np.abs(offset)) > cap:
            raise RuntimeError('Held arm command/feedback offset is too large for recovery')
        self.loaded_feedforward[8:15] = offset
        self.tracking_bias[8:15] = offset
        snapshot = self.restore_surface_view(snapshot, records)
        self.latest_snapshot = snapshot
        phases, planned = self._weighing_path(planned, snapshot,
                                             self.kinematics.fk(records[-1]['act_action']))
        opened = planned.copy()
        opened[15] = 0
        phases += [dict(name='settle_before_release', action=planned), dict(name='release_dwell', action=opened)]
        self._begin(action_vector(measured), 1, phases, target)

    def restore_regrasp(self, records, measured, snapshot, arm_command=None, wrist_roll_degrees=0.):
        """Pick the visible parcel on the observed pad, then verify a trial lift."""
        from scipy.spatial.transform import Rotation
        if records[-1]['sent_action'][15] > self.config['open_threshold']:
            raise RuntimeError('Regrasp source has not released its right-hand parcel')
        grasps = [r for r in records if r['cartesian_alignment'].get('state') == 'close_dwell'
                  and r['cartesian_alignment'].get('arm') == 1
                  and r['cartesian_alignment'].get('target') is not None]
        if not grasps:
            raise RuntimeError('No previous grasp geometry for weighing regrasp')
        grasp = grasps[-1]
        anchor = np.asarray(self.kinematics.fk(records[-1]['measured_state']).right_pose.pos_xyz)
        snapshot = self.restore_surface_view(snapshot, records)
        surfaces = snapshot.get('surfaces', [])
        if not surfaces:
            raise RuntimeError('Weighing pad is not visible for regrasp')
        surface = min(surfaces, key=lambda s: np.linalg.norm(np.asarray(s['position_base_m'])[:2] - anchor[:2]))
        self.weighing_target = np.asarray(surface['position_base_m'])
        candidates = [c for c in snapshot['candidates'] if np.linalg.norm(
            np.asarray(c['position_base_m'])[:2] - self.weighing_target[:2]) < .12]
        if len(candidates) != 1:
            raise RuntimeError(f'Expected one observed parcel on weighing pad, found {len(candidates)}')
        target = dict(candidates[0])
        planned = np.clip(action_vector(measured), self.low, self.high)
        planned[[7, 15]] = 0
        reference = self.kinematics.fk(planned)
        reference.right_pose.quat_xyzw = self.kinematics.fk(grasp['measured_state']).right_pose.quat_xyzw
        major = target.get('major_axis_xy')
        if major is not None:
            rotation = Rotation.from_quat(reference.right_pose.quat_xyzw)
            forward = -rotation.as_matrix()[:2, 2]
            major = np.asarray(major)
            if forward @ major < 0:
                major = -major
            yaw = np.arctan2(major[1], major[0]) - np.arctan2(forward[1], forward[0])
            reference.right_pose.quat_xyzw = (Rotation.from_euler('z', yaw) * rotation).as_quat().tolist()
        if wrist_roll_degrees:
            reference.right_pose.quat_xyzw = (Rotation.from_quat(reference.right_pose.quat_xyzw)
                * Rotation.from_euler('z', wrist_roll_degrees, degrees=True)).as_quat().tolist()
        point = self._grasp_point(target, reference.right_pose)
        old_pose = self.kinematics.fk(grasp['measured_state']).right_pose
        old_plane = grasp.get('rgbd_table_z_base_m')
        if old_plane is None and grasp.get('rgbd_surfaces'):
            old_plane = float(np.median([s['position_base_m'][2] for s in grasp['rgbd_surfaces']]))
        if old_plane is not None:
            point[2] = snapshot['table_z_base_m'] + old_pose.pos_xyz[2] - old_plane
        if point[2] < snapshot['table_z_base_m'] + .035:
            raise RuntimeError('Regrasp would put fingers too near the table')
        target['grasp_point_base_m'] = point.tolist()
        offset = np.asarray(records[-1]['sent_action'] if arm_command is None else arm_command)[8:15] - planned[8:15]
        if np.max(np.abs(offset)) > self.config['max_loaded_feedforward_rad']:
            raise RuntimeError('Empty-arm offset too large before regrasp')
        self.loaded_feedforward[8:15] = offset
        self.tracking_bias[8:15] = offset
        self.latest_snapshot = snapshot
        phases, planned = self._pickup_path(planned, measured, reference, 1, point, target)
        names = {'lift_before_alignment': 'lift_before_regrasp',
                 'translate_above_parcel': 'align_above_weighing_parcel',
                 'descend_before_close': 'descend_before_regrasp',
                 'insert_before_close': 'insert_before_regrasp',
                 'prepare_grasp_orientation': 'prepare_grasp_orientation',
                 'orient_before_descent': 'orient_before_descent'}
        for phase in phases:
            phase['name'] = names[phase['name']]
        planned[15] = self.config['closed_command']
        phases.append(dict(name='close_dwell', action=planned.copy()))
        # Interpolation and joint speed limits smooth the complete upward
        # goal, so the load can transfer without a low waypoint blocking it.
        lifted = self._solve(reference, 1, point + [0, 0, .08], planned)
        lifted[15] = self.config['closed_command']
        phases += [dict(name='regrasp_trial_lift', action=lifted),
                   dict(name='verify_regrasp', action=lifted.copy())]
        self.manipulation_kind = 'weighing_regrasp'
        self.ready_to_close[1] = False
        self._begin(action_vector(measured), 1, phases, target)

    def restore_pickup(self, records, measured, snapshot):
        """Retry an interrupted, still-open pickup using fresh parcel geometry."""
        rows = [r for r in records if r['cartesian_alignment'].get('arm') == 1
                and r['cartesian_alignment'].get('target') is not None]
        if not rows or records[-1]['sent_action'][15] > self.config['open_threshold']:
            raise RuntimeError('Pickup recovery requires a target and an open right gripper')
        old_target = np.asarray(rows[-1]['cartesian_alignment']['target']['position_base_m'])
        candidates = sorted(snapshot['candidates'], key=lambda c: np.linalg.norm(
            np.asarray(c['position_base_m'])[:2] - old_target[:2]))
        if not candidates or np.linalg.norm(np.asarray(candidates[0]['position_base_m'])[:2] - old_target[:2]) > .04:
            raise RuntimeError('Original pickup parcel is not visible at its observed location')
        raw = action_vector(rows[-1]['act_action'])
        poses = self.kinematics.fk(raw)
        target = dict(candidates[0])
        point = self._grasp_point(target, poses.right_pose)
        target['grasp_point_base_m'] = point.tolist()
        self.latest_snapshot = snapshot
        phases, planned = self._pickup_path(raw, measured, poses, 1, point, target)
        planned[15] = self.config['closed_command']
        phases.append(dict(name='close_dwell', action=planned.copy()))
        self.ready_to_close[1] = False
        self._begin(raw, 1, phases, target)

    def _pickup_path(self, raw, measured, poses, side, point, target):
        from scipy.spatial.transform import Rotation
        grip, start = (7, 0) if side == 0 else (15, 8)
        seed = np.clip(action_vector(measured), self.low, self.high)
        seed[grip] = 0
        actual = self.kinematics.fk(seed)
        feedback = np.asarray(self.pose(actual, side).pos_xyz)
        top = target.get('top_z_base_m', point[2] + .04)
        clearance = max(feedback[2], point[2] + self.config['clearance_m'], top + .045)
        if feedback[2] < top + .045:
            clearance = max(clearance, feedback[2] + .08)
        lift = feedback.copy(); lift[2] = clearance
        lifted = self._solve(actual, side, lift, seed)
        lifted[grip] = 0
        phases = [dict(name='lift_before_alignment', action=lifted.copy())]
        seed = lifted
        forward = -Rotation.from_quat(self.pose(poses, side).quat_xyzw).as_matrix()[:2, 2]
        pregrasp = point.copy()
        side_approach = np.linalg.norm(forward) > .65 and target.get('extent_xy_m') is not None
        if side_approach:
            forward /= np.linalg.norm(forward)
            pregrasp[:2] -= forward * (self.config['side_grasp_insertion_m'] + .06)
        plane = self.latest_snapshot.get('table_z_base_m') if self.latest_snapshot else None
        rotation_height = max(top + .12, point[2] + .14,
                              plane + .24 if plane is not None else point[2] + .14)
        # Position the open hand outside the end face before rotating it. A
        # high, fixed Cartesian roll unnecessarily consumes wrist range.
        above = np.r_[pregrasp[:2], max(clearance, rotation_height)]
        segment, seed = self._path('translate_above_parcel', actual, side, above, seed, grip)
        phases.extend(segment)
        rotation_point = np.r_[pregrasp[:2], rotation_height]
        for name, goal, reference in (
                ('prepare_grasp_orientation', rotation_point, actual),
                ('orient_before_descent', rotation_point, poses),
                ('descend_before_close', pregrasp, poses)):
            segment, seed = self._path(name, reference, side, goal, seed, grip)
            phases.extend(segment)
        if side_approach:
            segment, seed = self._path('insert_before_close', poses, side, point, seed, grip)
            phases.extend(segment)
        return phases, seed

    def camera_target(self, measured):
        if self.pending is None:
            return None
        p = self.pending
        phase = p['phases'][p['index']]
        poses = self.kinematics.fk(measured)
        side = p['side']
        if self.manipulation_kind == 'weighing_regrasp' and self.weighing_target is not None:
            point = np.asarray(p['target']['position_base_m'])
            if self.closed[1] and self.held_center_in_tool[1] is not None:
                from scipy.spatial.transform import Rotation
                point = np.asarray(poses.right_pose.pos_xyz) + Rotation.from_quat(
                    poses.right_pose.quat_xyzw).apply(self.held_center_in_tool[1])
            return .55 * (self.weighing_target + [0, 0, .05]) + .45 * point
        if (side == 1 and phase['action'][7] < self.config['other_arm_closed_threshold']
                and phase['name'] in ('stabilize_loaded_hold', 'lift_before_placement', 'center_above_weighing',
                                      'lower_before_release', 'settle_before_release', 'release_dwell')
                and self.weighing_target is not None):
            pose = poses.right_pose
            point = np.asarray(pose.pos_xyz)
            if self.held_center_in_tool[1] is not None:
                from scipy.spatial.transform import Rotation
                point = point + Rotation.from_quat(pose.quat_xyzw).apply(self.held_center_in_tool[1])
            return .6 * self.weighing_target + .4 * point
        if p['target'] is not None:
            return np.asarray(p['target']['position_base_m'])
        return np.asarray(self.pose(poses, side).pos_xyz)

    def _advance(self):
        p = self.pending
        p.update(index=p['index'] + 1, steps=0, stable=0, last_position=None,
                 load_stable=0, load_adjustments=0)
        if p['index'] == len(p['phases']):
            self.pending = None

    def _pending_action(self, measured):
        p = self.pending
        side = p['side']
        grip, start = (7, 0) if side == 0 else (15, 8)
        phase = p['phases'][p['index']]
        result = phase['action'].copy()
        p['steps'] += 1
        state = phase['name']
        info = dict(state=state, arm=side, phase_steps=p['steps'], target=p['target'])
        info['manipulation_kind'] = self.manipulation_kind
        if state == 'verify_regrasp':
            result[start:start + 7] += self.tracking_bias[start:start + 7]
            height = self.pose(self.kinematics.fk(measured), side).pos_xyz[2]
            grip_still = abs(measured[grip] - p.get('last_grip', measured[grip])) < .015
            p['last_grip'] = measured[grip]
            held = .20 < measured[grip] < .85 and grip_still and height >= self.grasp_z[side] + .05
            p['stable'] = p['stable'] + 1 if held else 0
            info.update(grasp_verified=bool(p['stable'] >= 6), stable_steps=p['stable'],
                        measured_gripper=float(measured[grip]), measured_tool_height_m=float(height))
            if p['stable'] >= 6:
                self._advance()
            elif p['steps'] >= 25:
                raise RuntimeError('Weighing regrasp not confirmed by finger contact and measured lift')
            return np.clip(result, self.low, self.high), info
        if state == 'stabilize_loaded_hold':
            actual = self.pose(self.kinematics.fk(measured), side)
            position = np.asarray(actual.pos_xyz)
            motion = (np.linalg.norm(position - p['last_position'])
                      if p['last_position'] is not None else float('inf'))
            angle_motion = (orientation_error(actual.quat_xyzw, p['last_orientation'])
                            if 'last_orientation' in p else float('inf'))
            p['last_position'] = position
            p['last_orientation'] = actual.quat_xyzw
            stationary = motion <= self.config['settle_motion_m'] and angle_motion <= .015
            p['stable'] = p['stable'] + 1 if stationary else 0
            info.update(stable_steps=p['stable'], motion_m=float(motion), orientation_motion_rad=float(angle_motion))
            if p['steps'] >= 10 and p['stable'] >= self.config['stable_steps']:
                offset = result[start:start + 7] - measured[start:start + 7]
                if np.max(np.abs(offset)) > self.config.get('max_loaded_feedforward_rad', .60):
                    raise RuntimeError('Static loaded hold offset is too large; keeping parcel grasped')
                self.loaded_feedforward[start:start + 7] = offset
                self.tracking_bias[start:start + 7] = offset
                seed = np.clip(action_vector(measured), self.low, self.high)
                seed[grip] = self.config['closed_command']
                if self.latest_snapshot is None:
                    raise RuntimeError('No fresh RGB-D for placement after loaded hold')
                phases, planned = self._weighing_path(seed, self.latest_snapshot,
                                                     self.kinematics.fk(self.held_action))
                opened = planned.copy()
                opened[grip] = 0
                phases += [dict(name='settle_before_release', action=planned),
                           dict(name='release_dwell', action=opened)]
                p.update(phases=phases, index=0, steps=0, stable=0, last_position=None)
                info['loaded_feedforward_rad'] = offset.tolist()
            elif p['steps'] >= self.config['alignment_timeout_steps']:
                raise RuntimeError('Held arm did not become stationary; keeping parcel grasped')
            return np.clip(result, self.low, self.high), info
        if state in ('close_dwell', 'release_dwell'):
            result[start:start + 7] += self.tracking_bias[start:start + 7]
            threshold = self.config[state + '_steps']
            if p['steps'] >= threshold:
                if state == 'close_dwell':
                    self.closed[side] = True
                    # Only local final-grasp residual survives policy resumption.
                    self.offset[start:start + 7] = result[start:start + 7] - self.held_action[start:start + 7]
                    self.grasp_z[side] = float(self.pose(self.kinematics.fk(phase['action']), side).pos_xyz[2])
                    if p['target'] is not None:
                        from scipy.spatial.transform import Rotation
                        actual = self.pose(self.kinematics.fk(measured), side)
                        self.held_place_orientation[side] = np.asarray(actual.quat_xyzw)
                        self.held_extent[side] = p['target'].get('extent_xy_m')
                        if p['target'].get('major_axis_xy') is not None:
                            self.held_major_axis_in_tool[side] = Rotation.from_quat(actual.quat_xyzw).inv().apply(
                                [*p['target']['major_axis_xy'], 0])
                        if self.latest_snapshot is not None and 'table_z_base_m' in self.latest_snapshot:
                            self.grasp_tool_height[side] = float(actual.pos_xyz[2] - self.latest_snapshot['table_z_base_m'])
                        self.held_center_in_tool[side] = Rotation.from_quat(actual.quat_xyzw).inv().apply(
                            np.asarray(p['target']['position_base_m']) - actual.pos_xyz)
                else:
                    self.closed[side] = False
                    self.ready_to_close[side] = True
                    self.offset[start:start + 7] = 0
                    self.grasp_z[side] = None
                    self.held_center_in_tool[side] = None
                    self.held_place_orientation[side] = None
                    self.held_major_axis_in_tool[side] = None
                    self.grasp_tool_height[side] = None
                    self.held_extent[side] = None
                    self.loaded_feedforward[start:start + 7] = 0
                    self.previous_grip[side] = 0.0
                self._advance()
            return np.clip(result, self.low, self.high), info
        feedback = self.pose(self.kinematics.fk(measured), side)
        desired = self.pose(self.kinematics.fk(result), side)
        position = np.asarray(feedback.pos_xyz)
        goal = np.asarray(desired.pos_xyz)
        xy_error = float(np.linalg.norm(position[:2] - goal[:2]))
        z_error = float(abs(position[2] - goal[2]))
        angle = orientation_error(feedback.quat_xyzw, desired.quat_xyzw)
        motion = (float(np.linalg.norm(position - p['last_position']))
                  if p['last_position'] is not None else float('inf'))
        tolerance = self.config['release_tolerance_m'] if state == 'settle_before_release' else self.config['alignment_tolerance_m']
        settled = (xy_error <= tolerance and z_error <= self.config['alignment_z_tolerance_m']
                   and angle <= self.config['orientation_tolerance_rad']
                   and motion <= self.config['settle_motion_m'])
        p['last_position'] = position
        p['stable'] = p['stable'] + 1 if settled else 0
        info.update(xy_error_m=xy_error, z_error_m=z_error, orientation_error_rad=angle,
                    max_joint_tracking_error_rad=float(np.max(np.abs(result[start:start+7] - measured[start:start+7]))),
                    stable_steps=p['stable'])
        # Refresh the load estimate only during stationary, high, near-target
        # holds. Never learn additional force at the table/contact descent.
        trial_lift = state == 'regrasp_trial_lift'
        open_lift = not self.closed[side] and state in ('lift_before_alignment', 'lift_before_regrasp', 'tracking_probe', 'translate_above_parcel', 'align_above_weighing_parcel', 'prepare_grasp_orientation', 'orient_before_descent')
        empty_descent = False
        if (not self.closed[side] and state in ('descend_before_close', 'descend_before_regrasp')
                and p['target'] is not None and self.latest_snapshot is not None):
            from scipy.spatial.transform import Rotation
            forward = -Rotation.from_quat(feedback.quat_xyzw).as_matrix()[:2, 2]
            norm = np.linalg.norm(forward)
            plane = self.latest_snapshot.get('table_z_base_m')
            if norm > .65 and plane is not None:
                forward /= norm
                grasp = np.asarray(p['target']['grasp_point_base_m'])
                empty_descent = ((grasp[:2] - position[:2]) @ forward >= .065
                                 and position[2] - plane >= .055)
        open_lift = open_lift or empty_descent
        free_descent = (state == 'lower_before_release' and self.closed[side]
                        and self.weighing_target is not None and self.grasp_tool_height[side] is not None
                        and position[2] - self.grasp_tool_height[side] - self.weighing_target[2] > .03)
        can_estimate = (position[2] > self.weighing_target[2] + .18 if self.weighing_target is not None else False)
        if free_descent:
            can_estimate = True
        if trial_lift:
            can_estimate = position[2] >= self.grasp_z[side] - .02 and angle < .45
        if empty_descent:
            can_estimate = True
        elif open_lift and p['target'] is not None:
            can_estimate = position[2] >= p['target'].get('top_z_base_m', p['target']['position_base_m'][2] + .04) + .025
        if ((self.closed[side] or open_lift) and self.last_sent is not None
                and state in ('lift_before_placement', 'center_above_weighing', 'regrasp_trial_lift', 'lift_before_alignment', 'lift_before_regrasp', 'lower_before_release', 'tracking_probe', 'translate_above_parcel', 'align_above_weighing_parcel', 'prepare_grasp_orientation', 'orient_before_descent', 'descend_before_close', 'descend_before_regrasp')
                and can_estimate and xy_error < (.04 if trial_lift else .025)
                and z_error < (.05 if trial_lift or can_estimate else .025) and p['steps'] >= 12):
            angle_motion = (orientation_error(feedback.quat_xyzw, p['load_orientation'])
                            if 'load_orientation' in p else float('inf'))
            p['load_orientation'] = feedback.quat_xyzw
            p['load_stable'] = p.get('load_stable', 0) + 1 if motion <= .002 and angle_motion <= .015 else 0
            if p['load_stable'] >= 8 and p.get('load_adjustments', 0) < (8 if trial_lift or open_lift or free_descent else 5):
                estimate = self.last_sent[start:start + 7] - measured[start:start + 7]
                increment = .08 if trial_lift or open_lift else .04
                change = np.clip(estimate - self.loaded_feedforward[start:start + 7], -increment, increment)
                cap = self.config['max_loaded_feedforward_rad']
                self.loaded_feedforward[start:start + 7] = np.clip(
                    self.loaded_feedforward[start:start + 7] + change, -cap, cap)
                p['load_stable'] = 0
                p['load_adjustments'] = p.get('load_adjustments', 0) + 1
                info['load_estimate_updated'] = True
        info['loaded_feedforward_rad'] = self.loaded_feedforward[start:start + 7].tolist()
        # Bounded proportional feedback, not an integral of a stalled error.
        # Wait for the nominal interpolation to settle before compensation.
        if p['steps'] >= 5:
            error_q = result[start:start + 7] - measured[start:start + 7]
            known_load = self.closed[side] or np.any(self.loaded_feedforward[start:start + 7])
            cap = (self.config['max_loaded_tracking_compensation_rad'] if known_load
                   else self.config['max_tracking_compensation_rad'])
            gain = (self.config['loaded_tracking_compensation_gain'] if self.closed[side]
                    else self.config['tracking_compensation_gain'])
            desired_bias = np.clip(self.loaded_feedforward[start:start + 7] + gain * error_q, -cap, cap)
            alpha = (self.config['loaded_tracking_compensation_alpha'] if self.closed[side]
                     else self.config['tracking_compensation_alpha'])
            self.tracking_bias[start:start + 7] += alpha * (desired_bias - self.tracking_bias[start:start + 7])
        result[start:start + 7] = np.clip(result[start:start + 7] + self.tracking_bias[start:start + 7],
                                         self.low[start:start + 7], self.high[start:start + 7])
        info['tracking_compensation_rad'] = self.tracking_bias[start:start + 7].tolist()
        if p['stable'] >= self.config['stable_steps']:
            self._advance()
        elif p['steps'] >= (self.config.get('loaded_alignment_timeout_steps', self.config['alignment_timeout_steps'])
                            if self.closed[side] or np.any(self.loaded_feedforward[start:start + 7]) else self.config['alignment_timeout_steps']):
            raise RuntimeError(f"{state} timed out: XY={xy_error:.3f} m, Z={z_error:.3f} m, orientation={angle:.3f} rad; gripper transition withheld")
        return result, info

    def apply(self, raw, measured, snapshot):
        raw, measured = action_vector(raw), action_vector(measured, 'measured state')
        self.latest_snapshot = snapshot
        self._remember_targets(snapshot)
        if self.pending is not None:
            return self._pending_action(measured)
        # Return to the original ACT trajectory once the parcel is clear of the
        # surface. Carrying a constant grasp joint residual distorts placement.
        for side, grip in enumerate((7, 15)):
            start = 0 if side == 0 else 8
            if self.grasp_z[side] is not None and np.any(self.offset[start:start+7]):
                height = self.pose(self.kinematics.fk(raw + self.offset), side).pos_xyz[2]
                if height - self.grasp_z[side] > self.config['lift_before_offset_decay_m']:
                    self.offset[start:start+7] *= self.config['transport_offset_decay']
                    if np.max(np.abs(self.offset[start:start+7])) < 1e-4:
                        self.offset[start:start+7] = 0
            if raw[grip] < self.config['open_threshold'] and not self.closed[side]:
                self.ready_to_close[side] = True
        closing = [side for side, grip in enumerate((7, 15))
                   if self.ready_to_close[side] and self.previous_grip[side] < self.config['close_threshold'] <= raw[grip]]
        releasing = [side for side, grip in enumerate((7, 15))
                     if self.closed[side] and raw[grip] <= self.config['open_threshold']]
        self.previous_grip = [float(raw[7]), float(raw[15])]
        planned = self._hold(raw + self.offset)
        if releasing:
            side = releasing[0]
            grip = (7, 15)[side]
            phases = []
            if (side == 1 and raw[7] < self.config['other_arm_closed_threshold']
                    and snapshot is not None and 'table_z_base_m' in snapshot):
                if self.last_sent is not None:
                    held = np.clip(self.last_sent, self.low, self.high)
                    held[grip] = self.config['closed_command']
                    if snapshot.get('surfaces'):
                        position = self.kinematics.fk(measured).right_pose.pos_xyz
                        surface = min(snapshot['surfaces'], key=lambda s: np.linalg.norm(
                            np.asarray(s['position_base_m'])[:2] - np.asarray(position)[:2]))
                        self.weighing_target = np.asarray(surface['position_base_m'])
                    self._begin(raw, side, [dict(name='stabilize_loaded_hold', action=held)])
                    return self._pending_action(measured)
                reference = self.kinematics.fk(planned)
                seed = planned.copy()
                seed[start:start + 7] = measured[start:start + 7]
                seed = np.clip(seed, self.low, self.high)
                phases, planned = self._weighing_path(seed, snapshot, reference)
            opened = planned.copy()
            opened[grip] = 0.0
            phases += [dict(name='settle_before_release', action=planned),
                       dict(name='release_dwell', action=opened)]
            self._begin(raw, side, phases)
            return self._pending_action(measured)
        if not closing:
            return planned, dict(state='act', offset_rad=self.offset.tolist())
        if len(closing) != 1:
            raise RuntimeError('Simultaneous closing requests are unsupported')
        side = closing[0]
        grip, other_grip, start = (7, 15, 0) if side == 0 else (15, 7, 8)
        self.ready_to_close[side] = False
        passthrough = raw[other_grip] > self.config['other_arm_closed_threshold']
        skip = self.initial_grasps < self.config['skip_initial_grasps']
        if skip and not passthrough:
            self.initial_grasps += 1
        target = None
        phases = []
        if not passthrough and not skip:
            if snapshot is None or not self._targets():
                raise RuntimeError('No valid RGB-D parcel near closing request; not closing')
            poses = self.kinematics.fk(planned)
            planned_pose = self.pose(poses, side)
            original = np.asarray(planned_pose.pos_xyz)
            candidates = sorted(self._targets(), key=lambda item: np.linalg.norm(
                self._grasp_point(item, planned_pose)[:2] - original[:2]))
            target = candidates[0]
            position = self._grasp_point(target, planned_pose)
            target = {**target, 'grasp_point_base_m': position.tolist()}
            distance = float(np.linalg.norm(position[:2] - original[:2]))
            if distance > min(self.config['max_target_distance_m'], self.config['max_xy_correction_m']) or abs(position[2] - original[2]) > 0.16:
                raise RuntimeError(f'Parcel is too far from ACT grasp ({distance:.3f} m XY); not closing')
            if len(candidates) > 1 and np.linalg.norm(self._grasp_point(candidates[1], planned_pose)[:2] - original[:2]) - distance < 0.02:
                raise RuntimeError('Closing target is ambiguous between parcels; not closing')
            phases, planned = self._pickup_path(raw, measured, poses, side, position, target)
        else:
            planned[grip] = 0.0
            phases.append(dict(name='settle_before_close', action=planned.copy()))
        planned[grip] = self.config['closed_command']
        phases.append(dict(name='close_dwell', action=planned.copy()))
        self._begin(raw, side, phases, target)
        result, info = self._pending_action(measured)
        info['grasp_kind'] = 'handover' if passthrough else ('initial' if skip else 'rgbd_pickup')
        return result, info
