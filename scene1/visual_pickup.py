"""Select remaining parcels and transport them to the observed weighing pad.

Only RGB-D and measured robot state are used. Local grasp, weighing, settlement
and regrasp remain owned by the established ResidualAssist implementation.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from .motion import action_vector, JOINTS
from .visual_cycle import VisualCycle, cloud
from .weighing_pose import alternate_weighing_pose, regrasp_posture_reserve
from .table_reference import table_frame, table_motion, move_target, table_boundary
from .receiver import NearFaceProjector
from .approach_edges import roof_edges, roof_edge_motion
from .clearance import open_fingers_clear
from .placement_height import PlacementHeight


def table_hull(snapshot):
    points, hsv = cloud(snapshot)
    selected = ((hsv[:, 0] >= 40) & (hsv[:, 0] <= 85) & (hsv[:, 1] >= 100)
                & (abs(points[:, 2]-snapshot['table_z_base_m']) < .012))
    points = points[selected, :2]
    return cv2.convexHull(points.astype(np.float32)) if len(points) >= 100 else None


def eligible_targets(snapshot, template, pad, rim):
    hull = table_hull(snapshot)
    if hull is None:
        return []
    result = []
    for candidate in snapshot['candidates']:
        xy = np.asarray(candidate['position_base_m'])[:2]
        size = np.r_[candidate.get('oriented_extent_m', [0., 0.]),
                     candidate['top_z_base_m']-snapshot['table_z_base_m']]
        ratio = size/np.asarray(template['extent'])
        # A shortened visible end shifts the centroid. Require an almost full
        # long axis before anchoring a new grasp, especially after interruption.
        if np.any(ratio < [.95,.85,.90]) or np.any(ratio > 1.25):
            continue
        if cv2.pointPolygonTest(hull, tuple(map(float, xy)), True) < .02:
            continue
        if np.linalg.norm(xy-np.asarray(pad)[:2]) < .14:
            continue
        if rim is not None and cv2.pointPolygonTest(rim['corners'], tuple(map(float, xy)), True) > -.025:
            continue
        result.append(copy.deepcopy(candidate))
    return result


def grasp_template(geometry, plan=None):
    result = dict(extent=np.asarray(geometry['extent']).tolist(),
                rotation_xyzw=geometry['rotation'].as_quat().tolist(),
                box_to_tool_m=(-geometry['rotation'].inv().apply(geometry['center'])).tolist())
    if plan is not None:
        result['box_to_tool_m'] = (np.asarray(plan['box_to_tool_m'])*np.asarray(geometry['extent'])/
                                   np.asarray(plan['extent'])).tolist()
        result['position_source'] = 'first_requested_grasp_target'
    return result


def grasp_plan(target, plane, point):
    axis = np.asarray(target['major_axis_xy']); axis /= np.linalg.norm(axis)
    rotation = np.column_stack(([*axis,0.],[-axis[1],axis[0],0.],[0.,0.,1.]))
    extent = np.r_[target['oriented_extent_m'], target['top_z_base_m']-plane]
    center = np.asarray(target['position_base_m']).copy(); center[2] = plane+extent[2]/2
    return dict(box_to_tool_m=(rotation.T@(np.asarray(point)-center)).tolist(), extent=extent.tolist())


def visible_pad(reference, snapshot):
    """Confirm a previously complete boundary under current hand occlusion.

    Partial observations must cover a substantial subset and support two
    nonparallel outer edges. They never shrink or relocate the stored boundary.
    """
    if reference is None:
        return False
    hull = np.asarray(reference['footprint_xy_m'], np.float32)
    rectangle = cv2.boxPoints(cv2.minAreaRect(hull))
    edges = np.roll(rectangle, -1, axis=0)-rectangle
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.column_stack((-edges[:,1], edges[:,0]))/lengths[:,None]
    for surface in snapshot.get('surfaces', []):
        if surface.get('points', 0) < 60 or abs(surface['position_base_m'][2]-reference['position_base_m'][2]) > .012:
            continue
        points = np.asarray(surface['footprint_xy_m'], np.float32)
        if cv2.contourArea(points) < .4*cv2.contourArea(hull):
            continue
        if any(cv2.pointPolygonTest(hull, tuple(map(float,p)), True) < -.02 for p in points):
            continue
        support = []
        for index, normal in enumerate(normals):
            on_edge = points[abs((points-rectangle[index])@normal) < .018]
            if len(on_edge) >= 2 and np.max(np.linalg.norm(on_edge[:,None]-on_edge[None,:], axis=2)) >= .06:
                support.append(index)
        if len(support) >= 2 and np.linalg.matrix_rank(normals[support], tol=.2) >= 2:
            return True
    return False


def update_pad_reference(reference, surfaces, pad):
    """Occlusion cannot replace a previously more complete support boundary."""
    candidates = [s for s in surfaces if not s.get('partial')
        and min(s['extent_xy_m']) >= .145
        and np.linalg.norm(np.asarray(s['position_base_m'])[:2]-np.asarray(pad)[:2]) < .05]
    if not candidates:
        return reference
    area = lambda s: cv2.contourArea(np.asarray(s['footprint_xy_m'],np.float32))
    best = max(candidates,key=area)
    if reference is None or area(best) > 1.05*area(reference):
        return copy.deepcopy(best)
    return reference


def restore_next_context(directory, cycle, assist, measured, ros_run_id):
    """Continue after completed parcels with two empty hands in the same scene."""
    directory = Path(directory)
    saved = json.loads((directory/'visual-cycle-checkpoint.json').read_text())
    metadata = json.loads((directory/'run.json').read_text())
    rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
    state = saved['controller']
    if not ros_run_id or saved['ros_run_id'] != ros_run_id:
        raise RuntimeError('Visual next source belongs to another ROS scene')
    if not state.get('done') or state.get('active') or not 0 < state['completed'] < cycle.config['max_parcels']:
        raise RuntimeError('Visual next source must have completed a parcel and returned home')
    previous_layout = metadata['config']['visual_cycle']
    completed = state['completed']
    if (not np.allclose(previous_layout.get('bin_lateral_offsets_m', [0.])[:completed],
                        cycle.config.get('bin_lateral_offsets_m', [0.])[:completed])
            or not np.allclose(previous_layout.get('bin_depth_offsets_m', [0.]*completed)[:completed],
                               cycle.config.get('bin_depth_offsets_m', [0.]*completed)[:completed])
            or not np.allclose(previous_layout.get('bin_tool_rpy_degrees', [70.,5.,-99.]),
                               cycle.config.get('bin_tool_rpy_degrees', [70.,5.,-99.]))):
        raise RuntimeError('Completed bin slots differ from the requested continuation layout')
    last = rows[-1]
    checkpoint_step = saved.get('step', last['step'])
    for following in rows:
        if following['step'] <= checkpoint_step:
            continue
        info = following['cartesian_alignment']
        if (info.get('strategy') != 'visual_pickup'
                or info.get('stage') not in ('select','clear','hover','descend')
                or max(following['sent_action'][7],following['sent_action'][15]) > .15):
            raise RuntimeError('Visual next source contains later manipulation beyond an empty approach')
    sent, expected = action_vector(last['sent_action']), action_vector(last['measured_state'])
    if max(measured[7], measured[15], sent[7], sent[15]) > .15:
        raise RuntimeError('Visual next requires two empty hands; held payloads need an exact visual weighing checkpoint')
    lo, hi = np.minimum(sent[JOINTS], expected[JOINTS])-.15, np.maximum(sent[JOINTS], expected[JOINTS])+.15
    if np.any(measured[JOINTS] < lo) or np.any(measured[JOINTS] > hi):
        raise RuntimeError('Visual next continuation physical pose differs from source')
    template = saved.get('pickup_template')
    if template is None:
        row = next(r for r in rows if r['cartesian_alignment'].get('state') == 'grasp_contact_confirmed')
        pose = assist.kinematics.fk(np.asarray(row['observation_state'])).right_pose
        target = min(row['rgbd_candidates'], key=lambda c: np.linalg.norm(np.asarray(c['position_base_m'])[:2]-np.asarray(pose.pos_xyz)[:2]))
        requested = next((r for r in reversed(rows[:rows.index(row)])
                          if r['cartesian_alignment'].get('grasp_target_base_m') is not None), None)
        if requested is None:
            raise RuntimeError('No requested grasp target in the continuation source')
        plan = grasp_plan(target, row['rgbd_table_z_base_m'], requested['cartesian_alignment']['grasp_target_base_m'])
        template = grasp_template(assist._geometry(target, pose, row['rgbd_table_z_base_m']), plan)
    cycle.home = np.asarray(state['home']['array'])
    cycle.completed = completed
    cycle.rim = {key: np.asarray(value['array'], dtype=value['dtype']) if isinstance(value, dict) and 'array' in value else value
                 for key, value in state['rim'].items()}
    if saved.get('weighing_pad') is not None:
        pad = np.asarray(saved['weighing_pad'])
    else:
        release = next(r for r in rows if r['cartesian_alignment'].get('state') == 'weighing_release')
        surfaces = [s for r in rows[:release['step']] for s in r['rgbd_surfaces']
                    if not s.get('partial') and min(s['extent_xy_m']) >= .145]
        tcp = assist.kinematics.matrix(np.asarray(release['observation_state'])[8:15], 1)[:3, 3]
        pad = np.asarray(min(surfaces, key=lambda s: np.linalg.norm(np.asarray(s['position_base_m'])[:2]-tcp[:2]))['position_base_m'])
    return template, pad, sent, None


class VisualPickup:
    def __init__(self, cycle, template, pad, measured, command, resume_lift=None):
        if resume_lift is not None:
            if command[15] < .9 or not .2 < measured[15] < .85 or max(command[7], measured[7]) > .15:
                raise RuntimeError('Lift continuation requires confirmed right-hand contact')
        elif max(measured[7], measured[15], command[7], command[15]) > .15:
            raise RuntimeError('Next visual pickup requires both hands empty and open')
        self.motion = VisualCycle(cycle.config, cycle.k, cycle.low, cycle.high, cycle.home)
        self.completed_parcels = cycle.completed
        self.motion.command = action_vector(command)
        self.motion.command[[7, 15]] = 0.
        self.motion.side = 1
        self.motion.goal = [None, None]
        self.template, self.pad, self.rim = template, np.asarray(pad), cycle.rim
        self.active, self.ready = True, False
        self.target = None
        self.last_stamp = None
        self.selected_frames = self.missing = self.lost = 0
        self.selection_head = None
        self.pad_reference_head = None
        self.table_reference = None
        self.approach_edge_reference = None
        self.unverified_table = 0
        self.pose_search_attempted = False
        self.posture_reserve_attempted = False
        self.feedback = None
        self.surface_reference = None
        self.pad_frames = 0
        self.placement_height = PlacementHeight()
        self._stage('select')
        if resume_lift is not None:
            self.target, self.plane = resume_lift['target'], resume_lift['plane']
            self.motion.geometry = resume_lift['geometry']
            self.motion.goal[1] = resume_lift['goal']
            self.motion.command[15] = .98
            self.surface_reference = resume_lift.get('surface')
            self.surface = self.surface_reference
            tool_rotation = Rotation.from_matrix(self.motion.goal[1][:3,:3])
            box_rotation = tool_rotation*self.motion.geometry['rotation']
            axis = box_rotation.as_matrix()[:,0]
            self.box_rotation = Rotation.from_euler('z', np.arctan2(axis[1], axis[0]))
            self.box_destination = self.motion.goal[1][:3,3]+tool_rotation.apply(self.motion.geometry['center'])
            self._stage(resume_lift.get('stage', 'lift'))
            self.pose_search_done=resume_lift.get('pose_search_done',False)
            self.pose_search_attempted=self.pose_search_done
            self.posture_reserve_attempted = resume_lift.get('posture_reserve_attempted', False)
            reference = resume_lift.get('posture_reserve_reference')
            self.posture_reserve_reference = None if reference is None else np.asarray(reference)
            if resume_lift.get('weighing_target_xy') is not None:
                self.weighing_target_xy=np.asarray(resume_lift['weighing_target_xy']).copy()

    def _stage(self, name):
        self.stage = name
        self.motion._stage(name)
        self.descent_blocked_steps = 0
        self.posture_reference = None
        self.posture_reserve_reference = None
        self.posture_reserve_frames = 0
        self.pose_search_done = False
        self.limit_frames = 0

    def camera_target(self, measured, assist):
        if self.stage == 'local_weighing':
            return assist.camera_target(measured)
        if self.stage == 'observe_pad':
            return self.pad
        if self.stage in ('lift', 'carry', 'prepare_weighing'):
            return self.motion.world_box(measured)[0]
        if self.target is not None:
            return np.asarray(self.target['position_base_m'])
        return None

    def _target_goal(self, measured, assist):
        major = np.asarray(self.target['major_axis_xy']); major /= np.linalg.norm(major)
        relative = Rotation.from_quat(self.template['rotation_xyzw'])
        current = self.motion.k.matrix(measured[8:15], 1)
        choices = []
        for sign in (1., -1.):
            axis = sign*major
            box = Rotation.from_matrix(np.column_stack(([*axis, 0.], [-axis[1], axis[0], 0.], [0., 0., 1.])))
            rotation = box*relative.inv()
            choices.append(rotation)
        rotation = min(choices, key=lambda r: (r*Rotation.from_matrix(current[:3, :3]).inv()).magnitude())
        pose = SimpleNamespace(pos_xyz=current[:3, 3], quat_xyzw=rotation.as_quat())
        goal = np.eye(4); goal[:3, :3] = rotation.as_matrix()
        goal[:3, 3] = assist._grasp_goal(self.target, pose, self.plane)
        self.approach_goal = goal.copy()
        if 'box_to_tool_m' in self.template:
            extent = np.r_[self.target['oriented_extent_m'], self.target['top_z_base_m']-self.plane]
            center = np.asarray(self.target['position_base_m']).copy()
            center[2] = self.plane+extent[2]/2
            offset = np.asarray(self.template['box_to_tool_m'])*extent/np.asarray(self.template['extent'])
            if self.motion.config.get('pickup_center_grasp_height', False):
                # The first ACT close may retain a local height trim. Later
                # visual pickups have their own observed top and support plane;
                # use their midpoint instead of inheriting ACT's trigger height.
                self.target['inherited_grasp_height_offset_m'] = float(offset[2])
                self.target['grasp_height_source'] = 'observed_vertical_midpoint'
                offset[2] = 0.
            approach_offset = offset.copy()
            extra = float(self.motion.config.get('pickup_extra_insertion_m', 0.))
            if not 0. <= extra <= .015:
                raise ValueError('Extra visual pickup insertion must be between 0 and 15 mm')
            applied = min(extra, max(0., abs(offset[0])-.015))
            offset[0] -= np.sign(offset[0])*applied
            self.target['extra_insertion_m'] = applied
            goal[:3,3] = center+(rotation*relative).apply(offset)
            self.approach_goal[:3,3] = center+(rotation*relative).apply(approach_offset)
            self.target['grasp_goal_xy'] = goal[:2,3].tolist()
        if self.motion.config.get('pickup_level_grasp', False):
            # Use the original regrasp's observed-axis orientation. Keep both
            # inherited position goals exactly, without propagating the first
            # contact's lateral finger tilt to every later parcel.
            forward = major if major@(np.asarray(self.target['position_base_m'])[:2]-goal[:2,3]) >= 0 else -major
            closing = np.r_[-forward[1], forward[0], 0.]
            # A small initial-contact pitch adjustment can leave more wrist
            # range for the later upright weighing pose. It does not change
            # the original regrasp pitch or either inherited position goal.
            offsets = self.motion.config.get('pickup_pitch_offsets_degrees')
            if offsets is None:
                pitch_offset = float(self.motion.config.get('pickup_pitch_offset_degrees', 0.))
            else:
                offsets = np.asarray(offsets, dtype=float)
                if (offsets.ndim != 1 or not 0 <= self.completed_parcels < len(offsets)
                        or not np.isfinite(offsets).all() or np.any((offsets < 0.) | (offsets > 5.))):
                    raise ValueError('Initial pickup pitch offsets need a bounded value for this parcel')
                pitch_offset = float(offsets[self.completed_parcels])
            if not 0. <= pitch_offset <= 5.:
                raise ValueError('Initial pickup pitch offset must be between 0 and 5 degrees')
            pitch = np.deg2rad(assist.wrist_config.get('regrasp_pitch_degrees', 12.5)+pitch_offset)
            z_axis = -np.r_[forward*np.cos(pitch), -np.sin(pitch)]
            goal[:3,:3] = np.column_stack((closing, np.cross(z_axis, closing), z_axis))
            self.approach_goal[:3,:3] = goal[:3,:3]
        return goal

    def _selection_view_ready(self, snapshot, head_info, fresh):
        """Anchor the grasp only after the camera has reached its target view.

        During a head turn the registered cloud can move by over a centimeter.
        Holding the arms here avoids comparing a home-view target with a moving
        camera estimate or using that stale target for the actual grasp.
        """
        if not fresh or head_info is None or head_info.get('state') != 'look_at_target':
            return False
        angles = snapshot.get('head_angles_rad')
        if angles is None:
            return False
        angles = np.rad2deg(np.asarray(angles, float))
        desired = np.asarray(head_info['desired_degrees'], float)
        command = np.asarray(head_info['command_degrees'], float)
        previous = self.selection_head
        self.selection_head = angles.copy()
        return bool(previous is not None and np.isfinite(angles).all()
            and np.max(abs(command-desired)) <= .1
            and np.max(abs(angles-desired)) <= 2.6
            and np.max(abs(angles-previous)) <= .25)

    def _pad_reference_view_ready(self, head_info, fresh):
        """Do not lock the largest pad footprint during a head turn."""
        if not fresh or head_info is None or head_info.get('state') != 'look_at_target':
            return False
        angles = np.asarray(head_info.get('measured_degrees', []), float)
        desired = np.asarray(head_info.get('desired_degrees', []), float)
        command = np.asarray(head_info.get('command_degrees', []), float)
        if any(value.shape != (2,) or not np.isfinite(value).all()
               for value in (angles, desired, command)):
            self.pad_reference_head = None
            return False
        previous = self.pad_reference_head
        self.pad_reference_head = angles.copy()
        return bool(previous is not None and np.max(abs(command-desired)) <= .1
            and np.max(abs(angles-desired)) <= 2.6
            and np.max(abs(angles-previous)) <= .25)

    def update(self, measured, snapshot, assist, head_info=None):
        measured = action_vector(measured)
        motion = self.motion
        if self.feedback is not None and np.array_equal(measured, self.feedback):
            self.frozen_steps += 1
            if self.frozen_steps >= 20 and max(abs(motion.command[JOINTS]-self.feedback_command[JOINTS])) > .01:
                raise RuntimeError('Visual pickup feedback frozen while arm commands change')
        else:
            self.feedback, self.feedback_command = measured.copy(), motion.command.copy()
            self.frozen_steps = 0
        motion.stage_steps += 1
        # Existing local phases retain their own separate bounded deadlines.
        if self.stage not in ('local_grasp', 'local_weighing') and motion.stage_steps > 500:
            raise RuntimeError('Visual pickup timed out at '+self.stage)
        info = dict(state='visual_pickup_'+self.stage, stage=self.stage,
                    strategy='visual_pickup', policy_controls_arms=False)
        if snapshot is None:
            self.missing += 1
            if self.missing >= 10:
                raise RuntimeError('Visual pickup lost registered RGB-D')
            return motion.command.copy(), info
        self.missing = 0
        fresh = snapshot['stamp'] != self.last_stamp
        self.last_stamp = snapshot['stamp']
        assist._remember(snapshot)
        reference_view_ready = self._pad_reference_view_ready(head_info, fresh)
        if reference_view_ready:
            self.surface_reference = update_pad_reference(self.surface_reference, snapshot.get('surfaces', []), self.pad)
        info['pad_reference_view_ready'] = reference_view_ready
        if self.stage in ('clear','hover','descend') and self.table_reference is not None:
            current=table_frame(table_hull(snapshot),snapshot['table_z_base_m'],table_boundary(snapshot))
            relative=table_motion(self.table_reference,current)
            if relative is None:
                if fresh:self.unverified_table+=1
                motion.servo_previous=[None,None];motion.stable=0
                info['table_reference_hold']=True
                if self.unverified_table>=30:
                    raise RuntimeError('Table reference could not be verified during visual approach')
                return motion.command.copy(),info
            self.unverified_table=0
            transform=relative['transform']
            self.target=move_target(self.anchor_target,transform)
            self.plane=self.anchor_plane+transform[2,3]
            self.grasp_goal=transform@self.anchor_grasp_goal
            self.approach_goal=transform@self.anchor_approach_goal
            if self.stage in ('hover','descend'):
                motion.goal[1]=self.approach_goal.copy()
                if self.stage=='hover':motion.goal[1][2,3]=self.target['top_z_base_m']+.10
                else:motion.goal[1][2,3]+=.005
            info['table_reference_motion']={k:v for k,v in relative.items() if k!='transform'}
        if self.stage in ('lift', 'observe_pad', 'carry', 'prepare_weighing'):
            self.lost = self.lost+1 if not .2 < measured[15] < .85 else 0
            if self.lost >= 4:
                raise RuntimeError('Visual pickup lost held parcel contact')
            if fresh:
                seen = assist._observe_box(snapshot, assist.kinematics.fk(measured))
                info['transport_visual_box'] = seen
                self.pose_frames = (getattr(self, 'pose_frames', 0)+1
                                    if seen is not None and seen['orientation_confirmed'] else 0)
                self.unverified_pose = (0 if seen is not None and seen['orientation_confirmed']
                                        else getattr(self, 'unverified_pose', 0)+1)
                if self.unverified_pose >= 30:
                    raise RuntimeError('Held parcel pose could not be verified during visual transport')
                motion.geometry = copy.deepcopy(assist.geometry[1])
                if self.pose_frames >= 3 and self.stage in ('lift', 'carry', 'prepare_weighing') and seen is not None and seen['orientation_confirmed']:
                    # Use the observed attachment transform while keeping a
                    # fixed upright box goal; a slipping parcel is not rigidly
                    # attached to the transform captured before liftoff.
                    rotation = self.box_rotation*motion.geometry['rotation'].inv()
                    motion.goal[1][:3,:3] = rotation.as_matrix()
                    motion.goal[1][:3,3] = self.box_destination-rotation.apply(motion.geometry['center'])
                    if self.stage == 'prepare_weighing':
                        gap = seen['bottom_base_m']-self.surface['position_base_m'][2]
                        current = motion.k.matrix(measured[8:15], 1)
                        motion.goal[1][2,3] = current[2,3]+np.clip(.03-gap, -.02, .02)
            if getattr(self, 'pose_frames', 0) < 3:
                motion.servo_previous = [None, None]
                motion.stable = 0
                info.update(pose_observation_hold=True, selected_target=self.target)
                return motion.command.copy(), info
        if self.stage == 'select':
            view_ready = self._selection_view_ready(snapshot, head_info, fresh)
            info['selection_view_ready'] = view_ready
            candidates = eligible_targets(snapshot, self.template, self.pad, self.rim)
            info['eligible_targets'] = len(candidates)
            if not candidates:
                self.selected_frames = 0
                return motion.command.copy(), info
            tcp = motion.k.matrix(measured[8:15], 1)[:3, 3]
            target = min(candidates, key=lambda c: np.linalg.norm(np.asarray(c['position_base_m'])[:2]-tcp[:2]))
            same = self.target is not None and np.linalg.norm(np.asarray(target['position_base_m'])[:2]-np.asarray(self.target['position_base_m'])[:2]) < .012
            if fresh:
                self.selected_frames = (self.selected_frames+1 if same else 1) if view_ready else 0
            self.target = target
            self.plane = snapshot['table_z_base_m']
            if self.selected_frames >= 3:
                self.grasp_goal = self._target_goal(measured, assist)
                self.table_reference=table_frame(table_hull(snapshot),self.plane,table_boundary(snapshot))
                self.approach_edge_reference = roof_edges(snapshot, self.target)
                self.anchor_target=copy.deepcopy(self.target)
                self.anchor_plane=self.plane
                self.anchor_grasp_goal=self.grasp_goal.copy()
                self.anchor_approach_goal=self.approach_goal.copy()
                info['table_reference_available']=self.table_reference is not None
                motion.goal[1] = motion.k.matrix(measured[8:15], 1)
                motion.goal[1][2, 3] = max(motion.goal[1][2, 3], max(c['top_z_base_m'] for c in candidates)+.12)
                self._stage('clear')
        elif self.stage == 'observe_pad':
            observed = visible_pad(self.surface_reference, snapshot)
            if fresh:
                self.pad_frames = self.pad_frames+1 if observed else 0
            info.update(pad_boundary_observed=observed, pad_stable_frames=self.pad_frames)
            if self.pad_frames >= 3:
                self.surface = copy.deepcopy(self.surface_reference)
                center, rotation, extent = motion.world_box(measured)
                destination = np.asarray(self.surface['position_base_m']).copy()
                destination[2] += extent[2]/2+.08
                self.box_destination = destination
                motion.goal[1] = motion.k.matrix(measured[8:15], 1)
                motion.goal[1][:3, 3] += destination-center
                self._stage('carry')
        elif self.stage in ('local_grasp', 'local_weighing'):
            if self.stage == 'local_weighing' and motion.config.get('weighing_live_support_height', False):
                height = self.placement_height.update(assist, snapshot,
                    view_ready=reference_view_ready, fresh=fresh)
                if height is not None:
                    self.surface = assist.pending['surface']
                    info['weighing_support_height'] = height
            # Keep all original regrasp checks. Only refine its lateral RGB
            # observation: a bright roof alone can bias the shaded near face.
            projector = getattr(assist, 'wrist_projector', None)
            if self.stage == 'local_weighing' and projector is not None:
                assist.wrist_projector = NearFaceProjector(projector)
            grasp_config = None
            tolerance = motion.config.get('pickup_grasp_z_tolerance_m')
            if self.stage == 'local_grasp' and tolerance is not None:
                tolerance = float(tolerance)
                if not 0. < tolerance <= assist.config['grasp_z_tolerance_m']:
                    raise ValueError('Visual pickup height tolerance must tighten the original grasp check')
                # Let the original grasp solver finish the height correction
                # before requesting closure. Preserve its phase clock, joint
                # limits and every other check; restore config after this call.
                grasp_config = assist.config
                assist.config = dict(grasp_config, grasp_z_tolerance_m=tolerance)
            try:
                command, local = assist.apply(assist.held_action, measured, snapshot)
            finally:
                if grasp_config is not None:
                    assist.config = grasp_config
                if projector is not None:
                    assist.wrist_projector = projector
            if (self.stage == 'local_weighing' and getattr(self, 'original_local_config', None) is not None
                    and (assist.pending is None or assist.pending['kind'] != 'weighing')):
                assist.config = self.original_local_config
                self.original_local_config = None
            motion.command = command.copy()
            info.update(local, strategy='visual_pickup', stage=self.stage, policy_controls_arms=False)
            if self.stage == 'local_grasp' and assist.first_grasp_complete and assist.pending is None:
                motion.geometry = copy.deepcopy(assist.geometry[1])
                center, rotation, extent = motion.world_box(measured)
                axis = rotation.as_matrix()[:,0]
                self.box_rotation = Rotation.from_euler('z', np.arctan2(axis[1], axis[0]))
                self.box_destination = center+np.array([0.,0.,.16])
                self.pose_frames = self.unverified_pose = 0
                motion.goal[1] = motion.k.matrix(measured[8:15], 1)
                motion.goal[1][2, 3] += .16
                self._stage('lift')
            elif self.stage == 'local_weighing' and assist.regrasp_complete and assist.pending is None:
                self.active, self.ready = False, True
        else:
            if self.stage in ('hover', 'descend'):
                displacement = assist._parcel_displacement(snapshot, self.target)
                # A hand can remove the near face while the other three roof
                # edges stay fixed. Compare those actual boundaries against
                # the selection frame, with the same table motion applied to
                # the target. Incomplete/inconsistent edges retain the old guard.
                edges = roof_edges(snapshot, self.target) if self.table_reference is not None else None
                edge_motion = roof_edge_motion(self.approach_edge_reference, edges)
                info['approach_component_displacement_m'] = displacement
                info['approach_edge_motion'] = edge_motion
                if edge_motion is not None:
                    displacement = edge_motion['displacement_m']
                info['approach_displacement_m'] = displacement
                if displacement is not None and displacement > .01:
                    raise RuntimeError('Selected parcel moved during visual open-hand approach')
            # Carry is a waypoint above the pad. The original weighing phase
            # still independently levels, fits the footprint and checks gap.
            posture = None
            posture_gain = .04
            if self.stage == 'prepare_weighing' and motion.config.get('weighing_posture_bias', False):
                if self.posture_reference is None:
                    goal = motion.goal[1].copy()
                    target_rotation = Rotation.from_matrix(goal[:3,:3])
                    def objective(q):
                        pose = motion.k.matrix(q, 1)
                        return np.r_[pose[:3,3]-goal[:3,3], .12*(target_rotation*
                            Rotation.from_matrix(pose[:3,:3]).inv()).as_rotvec()]
                    fit = least_squares(objective, np.clip(measured[8:15], motion.low[8:15]+1e-6,
                        motion.high[8:15]-1e-6), bounds=(motion.low[8:15], motion.high[8:15]), max_nfev=200)
                    error = objective(fit.x)
                    if np.linalg.norm(error[:3]) > .012 or np.linalg.norm(error[3:])/.12 > .14:
                        raise RuntimeError('No bounded robot posture for visual weighing preparation')
                    self.posture_reference = fit.x
                posture = [None, self.posture_reference]
                info['posture_reference_rad'] = self.posture_reference.tolist()
            if self.stage == 'prepare_weighing' and self.posture_reserve_reference is not None:
                posture = [None, self.posture_reserve_reference]
                posture_gain = .3
                info['posture_reserve_reference_rad'] = self.posture_reserve_reference.tolist()
            before_servo = motion.command.copy()
            preparation_tilt=motion.config.get('weighing_prepare_tilt_degrees',8.)
            if self.stage=='prepare_weighing' and not 0<preparation_tilt<=getattr(assist,'config',{}).get('level_target_degrees',10.):
                raise ValueError('Preparation tilt cannot exceed the original weighing level target')
            angle_limit=(.25 if self.stage=='carry' else np.deg2rad(preparation_tilt)
                if self.stage=='prepare_weighing' and 'weighing_prepare_tilt_degrees' in motion.config else None)
            descent_alignment = self.stage == 'descend' and motion.config.get('pickup_level_grasp', False)
            coupled_pose_request = (self.stage == 'prepare_weighing' and self.pose_search_done
                and motion.config.get('weighing_coupled_pose_request', False))
            arrived, errors = motion._arrived(measured, angle_limit=angle_limit,
                position_limit=.008 if self.stage == 'prepare_weighing' else None,
                posture_targets=posture, posture_gain=posture_gain,
                descent_alignment=descent_alignment, coupled_pose_request=coupled_pose_request)
            info['pose_errors'] = errors
            if self.stage == 'prepare_weighing':
                info['weighing_coupled_pose_request'] = bool(coupled_pose_request)
            if descent_alignment:
                # Apply the original regrasp sweep check to the same measured
                # open fingers before accepting this descent command.
                delta = motion.command[8:15]-before_servo[8:15]
                clear = all(open_fingers_clear(motion.k.matrix(measured[8:15]+a*delta, 1),
                            self.target, self.plane) for a in (0., .5, 1.))
                info['open_fingers_clear'] = bool(clear)
                self.descent_blocked_steps = 0 if clear else self.descent_blocked_steps+1
                if not clear:
                    motion.command = before_servo
                    motion.stable = 0
                    arrived = False
                    if self.descent_blocked_steps >= 20:
                        raise RuntimeError('Open finger sweep intersects observed parcel; pickup stopped')
            if self.stage == 'prepare_weighing':
                seen = info.get('transport_visual_box') or {}
                gap = seen.get('bottom_base_m', np.inf)-self.surface['position_base_m'][2]
                info['preparation_gap_m'] = gap if np.isfinite(gap) else None
                at_limit = np.any(np.minimum(motion.command[8:15]-motion.low[8:15],
                    motion.high[8:15]-motion.command[8:15]) < .002)
                # Arrival requires the full tool orientation as well as the
                # observed roof tilt. Either can remain blocked at a limit.
                orientation_blocked = (seen.get('tilt_degrees', 0.) > preparation_tilt
                    or any(e['angle_rad'] >= (angle_limit if angle_limit is not None else .14)
                           for e in errors))
                if fresh:
                    self.limit_frames = (self.limit_frames+1 if at_limit
                        and seen.get('orientation_confirmed') and orientation_blocked
                        and .022 <= gap <= .038 else 0)
                if (motion.config.get('weighing_pose_search',False) and not self.pose_search_done
                        and self.limit_frames >= 25):
                    self.pose_search_done = True
                    self.pose_search_attempted = True
                    alternative = alternate_weighing_pose(motion.k, measured, before_servo,
                        motion.geometry, motion.goal[1], self.surface)
                    # Planning may outlast a camera interval. Send the already
                    # held action and wait for fresh feedback before moving.
                    motion.command = before_servo
                    motion.servo_previous = [None,None]
                    motion.stable = 0
                    arrived = False
                    info['weighing_pose_search'] = dict(found=alternative is not None)
                    if alternative is not None:
                        self.box_destination[:2] = alternative['center'][:2]
                        self.box_rotation = alternative['rotation']
                        self.weighing_target_xy = alternative['center'][:2].copy()
                        info['weighing_pose_search'].update({k:v for k,v in alternative.items()
                            if k not in ('center','rotation')})
                        motion.servo_previous = [None,None]
                        motion.stable = 0
                        arrived = False
                arrived = (arrived and seen.get('orientation_confirmed', False)
                           and seen.get('tilt_degrees', 90.) <= preparation_tilt and .022 <= gap <= .038)
                if self.posture_reserve_reference is not None:
                    close = abs(motion.command[12]-self.posture_reserve_reference[4]) <= np.deg2rad(3)
                    if fresh:
                        self.posture_reserve_frames = self.posture_reserve_frames+1 if arrived and close else 0
                    arrived = arrived and self.posture_reserve_frames >= 3
                elif (arrived and motion.config.get('weighing_posture_reserve', False)
                      and self.pose_search_attempted and not self.posture_reserve_attempted
                      and motion.high[12]-before_servo[12] < np.deg2rad(5)):
                    self.posture_reserve_attempted = True
                    # Never reset this stage's 500-step clock or the later
                    # original weighing/regrasp reference. An accepted plan
                    # must be physically reached before the normal handoff.
                    plan = (regrasp_posture_reserve(motion.k, measured, before_servo, motion.goal[1])
                            if motion.stage_steps <= 420 else None)
                    info['weighing_posture_reserve'] = dict(accepted=plan is not None)
                    if plan is not None:
                        self.posture_reserve_reference = plan['reference']
                        info['weighing_posture_reserve'].update({key:value for key,value in plan.items()
                            if key != 'reference'})
                        motion.command = before_servo
                        motion.servo_previous = [None, None]
                        motion.stable = 0
                        arrived = False
            if arrived and (self.stage != 'carry' or getattr(self, 'pose_frames', 0) >= 3):
                if self.stage == 'clear':
                    motion.goal[1] = self.approach_goal.copy()
                    motion.goal[1][2, 3] = self.target['top_z_base_m']+.10
                    self._stage('hover')
                elif self.stage == 'hover':
                    motion.goal[1] = self.approach_goal.copy()
                    motion.goal[1][2, 3] += .005
                    self._stage('descend')
                elif self.stage == 'descend':
                    assist.offset[:] = 0.
                    orientation = (dict(grasp_rotation=Rotation.from_matrix(self.grasp_goal[:3,:3]))
                        if motion.config.get('pickup_level_grasp', False) else {})
                    assist._begin('grasp', 1, motion.command.copy(), target=self.target,
                                  plane=self.plane, grasp_z=float(self.grasp_goal[2, 3]), wrist_verified=False,
                                  **orientation)
                    self._stage('local_grasp')
                elif self.stage == 'lift':
                    support = motion._head_support(snapshot, measured)
                    info['lift_evidence'] = support
                    if (support is not None and support['bottom_m'] >= self.plane+.06
                            and getattr(self, 'pose_frames', 0) >= 3):
                        self._stage('observe_pad')
                elif self.stage == 'carry':
                    self._stage('prepare_weighing')
                elif self.stage == 'prepare_weighing':
                    assist.offset[:] = 0.
                    assist.weighing_target = np.asarray(self.surface['position_base_m'])
                    if hasattr(self,'weighing_target_xy'):
                        assist.weighing_target = assist.weighing_target.copy()
                        assist.weighing_target[:2] = self.weighing_target_xy
                    base = motion.command.copy(); base[15] = .98
                    self.original_local_config = assist.config
                    scale = motion.config.get('weighing_gain_scale', 1.)
                    assist.config = dict(assist.config, gain=assist.config['gain']*scale,
                        max_joint_step_rad=assist.config['max_joint_step_rad']*min(1., np.sqrt(scale)))
                    assist._begin('weighing', 1, base, stage='scan_left', surface=self.surface,
                                  reference_base=base.copy(), requested_open=0., hold_grip=.98)
                    self._stage('local_weighing')
        if self.target is not None:
            info['selected_target'] = self.target
        if getattr(self, 'surface', None) is not None:
            info['weighing_surface'] = self.surface
        info['goal_tcp'] = [g.tolist() if g is not None else None for g in motion.goal]
        if info['stage'] != self.stage:
            info['transition_to'] = self.stage
        return motion.command.copy(), info
