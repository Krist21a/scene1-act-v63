import json
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import eligible_targets, visible_pad, VisualPickup, grasp_plan, grasp_template, update_pad_reference
from scene1.table_reference import table_frame
from scene1.kinematics import UrdfArmKinematics


def boundary_samples(hull):
    return np.concatenate([np.linspace(a, b, 20) for a, b in zip(hull, np.roll(hull, -1, axis=0))])
from scene1.residual import ResidualAssist
from scene1.receiver import NearFaceProjector
from test_visual_cycle import Arm


class VisualPickupTests(unittest.TestCase):
    def initial_grasp_failure(self, enabled):
        import copy
        fixture = json.loads((Path(__file__).parent/'fixtures/initial_grasp_tilt_v48.json').read_text())
        arm = UrdfArmKinematics(description=fixture['public_urdf'])
        measured, command = np.array(fixture['measured']), np.array(fixture['previous_command'])
        motion = VisualCycle(dict(pickup_level_grasp=enabled, pickup_extra_insertion_m=.015),
            arm, np.full(16,-10.), np.full(16,10.), command)
        pickup = VisualPickup(motion, fixture['template'], self.pad, measured, command)
        pickup.target = copy.deepcopy(fixture['target']); pickup.plane = fixture['plane']
        assist = ResidualAssist(fixture['config'], arm, np.full(16,-10.), np.full(16,10.))
        assist.wrist_config = fixture['wrist_config']
        return fixture, pickup, assist, measured, command

    def test_level_grasp_keeps_real_failure_positions_and_uses_observed_parcel_axis(self):
        _, original, old_assist, measured, _ = self.initial_grasp_failure(False)
        fixture, pickup, assist, _, _ = self.initial_grasp_failure(True)
        old_goal = original._target_goal(measured, old_assist)
        goal = pickup._target_goal(measured, assist)
        np.testing.assert_array_equal(goal[:3,3], old_goal[:3,3])
        np.testing.assert_array_equal(pickup.approach_goal[:3,3], original.approach_goal[:3,3])
        np.testing.assert_allclose(goal[:3,3], fixture['point'], atol=1e-10)
        self.assertGreater(abs(old_goal[2,0]), np.sin(np.deg2rad(5)))
        self.assertAlmostEqual(goal[2,0], 0.)
        self.assertAlmostEqual(np.rad2deg(np.arcsin(goal[2,2])), 12.5)
        np.testing.assert_allclose(goal[:3,:3].T@goal[:3,:3], np.eye(3), atol=1e-12)
        toward = np.asarray(pickup.target['position_base_m'])[:2]-goal[:2,3]
        self.assertGreater(-goal[:2,2]@toward, 0.)
        self.assertEqual(pickup.target['extra_insertion_m'], .015)

    def test_real_preclose_feedback_cannot_close_until_original_orientation_guard_passes(self):
        for enabled in (False, True):
            fixture, pickup, assist, measured, command = self.initial_grasp_failure(enabled)
            goal = pickup._target_goal(measured, assist)
            fields = dict(grasp_rotation=Rotation.from_matrix(goal[:3,:3])) if enabled else {}
            assist._begin('grasp', 1, command, target=pickup.target, plane=pickup.plane,
                grasp_z=fixture['point'][2], wrist_verified=False, **fields)
            assist.pending['stable'] = 1  # the actual old closure was its second ready frame
            action, info = assist._pending_action(measured, None)
            if enabled:
                self.assertEqual(action[15], 0.)
                self.assertGreater(info['finger_line_tilt_degrees'], 5.)
                self.assertNotEqual(assist.pending.get('stage'), 'close_confirm')
                self.assertLessEqual(np.max(abs(action[8:15]-command[8:15])),
                    assist.config['max_joint_step_rad']+1e-10)
            else:
                self.assertEqual(action[15], .98)
                self.assertEqual(assist.pending['stage'], 'close_confirm')

    def test_initial_pitch_offset_keeps_grasp_positions_and_original_regrasp_pitch(self):
        for offset in (2.5, 5.):
            self.check_initial_pitch_offset(offset)

    def check_initial_pitch_offset(self, offset):
        fixture, pickup, assist, measured, _ = self.initial_grasp_failure(True)
        original = pickup._target_goal(measured, assist)
        approach = pickup.approach_goal.copy()
        original_wrist_config = assist.wrist_config.copy()
        pickup.motion.config['pickup_pitch_offset_degrees'] = offset
        adjusted = pickup._target_goal(measured, assist)
        np.testing.assert_array_equal(adjusted[:3,3], original[:3,3])
        np.testing.assert_array_equal(pickup.approach_goal[:3,3], approach[:3,3])
        np.testing.assert_allclose(adjusted[:3,3], fixture['point'], atol=1e-10)
        np.testing.assert_allclose(adjusted[:3,:3], original[:3,:3] @
            Rotation.from_euler('x', offset, degrees=True).as_matrix(), atol=1e-12)
        self.assertAlmostEqual(adjusted[2,0], 0.)
        self.assertAlmostEqual(np.rad2deg(np.arcsin(adjusted[2,2])), 12.5+offset)
        self.assertEqual(assist.wrist_config, original_wrist_config)

    def test_initial_pitch_adjustment_cannot_bypass_real_preclose_guard(self):
        fixture, pickup, assist, measured, command = self.initial_grasp_failure(True)
        pickup.motion.config['pickup_pitch_offset_degrees'] = 5.
        goal = pickup._target_goal(measured, assist)
        assist._begin('grasp', 1, command, target=pickup.target, plane=pickup.plane,
            grasp_z=fixture['point'][2], wrist_verified=False,
            grasp_rotation=Rotation.from_matrix(goal[:3,:3]))
        assist.pending['stable'] = 1
        action, info = assist._pending_action(measured, None)
        self.assertEqual(action[15], 0.)
        self.assertGreater(info['finger_line_tilt_degrees'], 5.)
        self.assertNotEqual(assist.pending.get('stage'), 'close_confirm')
        self.assertLessEqual(np.max(abs(action[8:15]-command[8:15])),
            assist.config['max_joint_step_rad']+1e-10)

    def test_initial_pitch_offset_rejects_unbounded_or_nonfinite_values(self):
        for value in (-.01, 5.01, float('nan'), float('inf')):
            _, pickup, assist, measured, _ = self.initial_grasp_failure(True)
            pickup.motion.config['pickup_pitch_offset_degrees'] = value
            with self.assertRaisesRegex(ValueError, 'Initial pickup pitch offset'):
                pickup._target_goal(measured, assist)

    def test_descend_handoff_enables_existing_grasp_guard_only_when_configured(self):
        for enabled in (False, True):
            _, pickup, assist, measured, _ = self.initial_grasp_failure(enabled)
            pickup.grasp_goal = pickup._target_goal(measured, assist)
            pickup._stage('descend')
            begun = []
            assist._begin = lambda *args, **kwargs: begun.append((args, kwargs))
            assist._parcel_displacement = lambda *args: 0.
            with patch.object(pickup.motion, '_arrived', return_value=(True, [])):
                pickup.update(measured, self.snapshot, assist)
            self.assertEqual(pickup.stage, 'local_grasp')
            self.assertEqual(begun[0][0][0], 'grasp')
            self.assertEqual('grasp_rotation' in begun[0][1], enabled)
            if enabled:
                np.testing.assert_allclose(begun[0][1]['grasp_rotation'].as_matrix(), pickup.grasp_goal[:3,:3])

    def test_near_face_measurement_is_scoped_to_post_weighing_regrasp(self):
        measured = np.zeros(16)
        for stage, fail in [('local_grasp', False), ('local_weighing', False), ('local_weighing', True)]:
            with self.subTest(stage=stage, fail=fail):
                cycle = VisualCycle({}, Arm(), np.full(16, -3.), np.full(16, 3.), measured)
                pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
                pickup._stage(stage)
                original = object()
                assist = SimpleNamespace(_remember=lambda snapshot: None,
                    wrist_projector=original, held_action=measured,
                    first_grasp_complete=False, regrasp_complete=False,
                    pending=dict(kind='pregrasp'))
                def apply(*args):
                    if stage == 'local_weighing':
                        self.assertIsInstance(assist.wrist_projector, NearFaceProjector)
                    else:
                        self.assertIs(assist.wrist_projector, original)
                    if fail:
                        raise RuntimeError('original collision check')
                    return measured.copy(), dict(state='small_pregrasp')
                assist.apply = apply
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'original collision check'):
                        pickup.update(measured, self.snapshot, assist)
                else:
                    command, _ = pickup.update(measured, self.snapshot, assist)
                    np.testing.assert_array_equal(command, measured)
                self.assertIs(assist.wrist_projector, original)

    def transport_assist(self, confirmed=False):
        geometry = dict(center=np.zeros(3), rotation=Rotation.identity(), extent=np.array([.11,.04,.1]))
        return SimpleNamespace(_remember=lambda snapshot: None,
            geometry=[None, geometry], kinematics=SimpleNamespace(fk=lambda measured: None),
            _observe_box=lambda *args: dict(orientation_confirmed=confirmed))

    def setUp(self):
        fixture = Path(__file__).parent/'fixtures'
        self.snapshot = dict(np.load(fixture/'next_parcel_rgbd.npz'))
        self.snapshot.update(json.loads((fixture/'next_parcel_candidates.json').read_text()), stamp=1.)
        self.template = dict(extent=[.112, .042, .103], rotation_xyzw=[0., 0., 0., 1.])
        self.pad = [.425, -.56, -.084]

    def test_real_failed_act_scene_keeps_three_parcels_excludes_robot(self):
        candidates = eligible_targets(self.snapshot, self.template, self.pad, None)
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(c['position_base_m'][0] > .3 for c in candidates))
        self.assertFalse(any(np.linalg.norm(np.asarray(c['position_base_m'])[:2]-[.33, -.30]) < .06 for c in candidates))

    def test_no_table_evidence_or_old_bin_object_cannot_be_selected(self):
        with patch('scene1.visual_pickup.table_hull', return_value=None):
            self.assertEqual(eligible_targets(self.snapshot, self.template, self.pad, None), [])
        rim = dict(corners=np.array([[.4,-.4],[.6,-.4],[.6,-.2],[.4,-.2]], np.float32))
        self.assertEqual(len(eligible_targets(self.snapshot, self.template, self.pad, rim)), 2)

    def test_occluded_shortened_parcel_cannot_anchor_a_new_grasp(self):
        import copy
        candidates=eligible_targets(self.snapshot,self.template,self.pad,None)
        self.assertEqual(len(candidates),3)
        short=copy.deepcopy(candidates[0]);short['oriented_extent_m'][0]*=.85
        snapshot=dict(self.snapshot,candidates=[short])
        self.assertEqual(eligible_targets(snapshot,self.template,self.pad,None),[])

    def test_selection_holds_arms_until_target_view_and_fresh_feedback_settle(self):
        measured=np.zeros(16)
        cycle=VisualCycle({},Arm(),np.full(16,-3.),np.full(16,3.),measured)
        pickup=VisualPickup(cycle,self.template,self.pad,measured,measured)
        assist=SimpleNamespace(_remember=lambda snapshot:None,
            _grasp_goal=lambda *args:np.array([.4,-.3,-.03]))
        head=dict(state='look_at_target',desired_degrees=[-33.,23.],command_degrees=[-21.,23.])
        self.snapshot['head_angles_rad']=np.deg2rad([-10.,24.5])
        # Reproduce the old failure: matching full candidates while the camera
        # is still far from its requested view must never start an approach.
        for _ in range(6):
            self.snapshot['stamp']+=.1
            action,info=pickup.update(measured,self.snapshot,assist,head)
            self.assertEqual(pickup.stage,'select');self.assertFalse(info['selection_view_ready'])
            np.testing.assert_array_equal(action,measured)
        head['command_degrees']=[-33.,23.]
        self.snapshot['head_angles_rad']=np.deg2rad([-33.,24.5])
        self.snapshot['stamp']+=.1
        pickup.update(measured,self.snapshot,assist,head)
        self.assertEqual(pickup.stage,'select')  # arrived, but still moving
        for _ in range(2):
            self.snapshot['stamp']+=.1
            pickup.update(measured,self.snapshot,assist,head)
            self.assertEqual(pickup.stage,'select')
        for _ in range(4):pickup.update(measured,self.snapshot,assist,head)
        self.assertEqual(pickup.stage,'select')  # duplicate images cannot count
        self.snapshot['stamp']+=.1
        pickup.update(measured,self.snapshot,assist,head)
        self.assertEqual(pickup.stage,'clear')
        np.testing.assert_array_equal(pickup.motion.command,measured)

    def test_selection_without_head_feedback_cannot_move_or_close(self):
        measured=np.zeros(16)
        cycle=VisualCycle({},Arm(),np.full(16,-3.),np.full(16,3.),measured)
        pickup=VisualPickup(cycle,self.template,self.pad,measured,measured)
        assist=SimpleNamespace(_remember=lambda snapshot:None)
        for _ in range(5):
            self.snapshot['stamp']+=.1
            command,_=pickup.update(measured,self.snapshot,assist)
            self.assertEqual(pickup.stage,'select')
            np.testing.assert_array_equal(command,measured)

    def table_pickup(self):
        measured=np.zeros(16)
        cycle=VisualCycle({},Arm(),np.full(16,-3.),np.full(16,3.),measured)
        pickup=VisualPickup(cycle,self.template,self.pad,measured,measured)
        pickup._stage('hover')
        hull=np.array([[.2,-.9],[1.,-.9],[1.,.5],[.2,.5]],np.float32)
        pickup.table_reference=table_frame(hull,-.08);pickup.anchor_plane=-.08
        pickup.anchor_target=dict(position_base_m=[.48,-.31,-.02],top_z_base_m=.02,
            major_axis_xy=[1.,0.],oriented_extent_m=[.11,.04],extent_xy_m=[.11,.04],
            grasp_goal_xy=[.45,-.31])
        pickup.anchor_grasp_goal=np.eye(4);pickup.anchor_grasp_goal[:3,3]=[.45,-.31,-.02]
        pickup.anchor_approach_goal=pickup.anchor_grasp_goal.copy()
        pickup.anchor_approach_goal[0,3]-=.01
        assist=SimpleNamespace(_remember=lambda snapshot:None,
            _parcel_displacement=ResidualAssist._parcel_displacement)
        return pickup,assist,hull,measured

    def test_observed_table_updates_both_goals_and_guard_without_accumulation(self):
        import copy
        pickup,assist,hull,measured=self.table_pickup()
        target=copy.deepcopy(pickup.anchor_target);target['position_base_m'][0]+=.012
        self.snapshot.update(table_z_base_m=-.08,candidates=[target],surfaces=[])
        with patch('scene1.visual_pickup.table_hull',return_value=hull+[.012,0.]), \
                patch('scene1.visual_pickup.table_boundary',return_value=boundary_samples(hull+[.012,0.])), \
                patch.object(pickup.motion,'_arrived',return_value=(False,[])):
            for _ in range(3):
                self.snapshot['stamp']+=.1
                command,info=pickup.update(measured,self.snapshot,assist)
                np.testing.assert_allclose(pickup.grasp_goal[:3,3],[.462,-.31,-.02],atol=1e-6)
                np.testing.assert_allclose(pickup.motion.goal[1][:3,3],[.452,-.31,.12],atol=1e-6)
                self.assertIn('table_reference_motion',info)
                np.testing.assert_array_equal(command,measured)
            self.snapshot['candidates'][0]['position_base_m'][0]+=.016
            with self.assertRaisesRegex(RuntimeError,'Selected parcel moved'):
                pickup.update(measured,self.snapshot,assist)

    def test_three_edges_keep_original_motion_limit_and_incomplete_evidence_falls_back(self):
        reference = {f'{axis}:{sign}': dict(position_m=sign*size/2)
                     for axis, size in enumerate([.11, .04]) for sign in [-1, 1]}
        for component, edge_shift, missing, stop in [(.012, 0., False, False),
                (0., .014, False, True), (.012, 0., True, True)]:
            with self.subTest(component=component, edge_shift=edge_shift, missing=missing):
                pickup, assist, hull, measured = self.table_pickup()
                pickup.approach_edge_reference = reference
                assist._parcel_displacement = lambda *args: component
                current = {key: dict(position_m=value['position_m']+
                                    (edge_shift if key.startswith('0:') else 0.))
                           for key, value in reference.items()}
                if missing:
                    current = {key: value for key, value in current.items() if key.startswith('0:')}
                self.snapshot.update(table_z_base_m=-.08, surfaces=[])
                with patch('scene1.visual_pickup.table_hull', return_value=hull), \
                        patch('scene1.visual_pickup.table_boundary', return_value=boundary_samples(hull)), \
                        patch('scene1.visual_pickup.roof_edges', return_value=current), \
                        patch.object(pickup.motion, '_arrived', return_value=(False, [])) as servo:
                    if stop:
                        with self.assertRaisesRegex(RuntimeError, 'Selected parcel moved'):
                            pickup.update(measured, self.snapshot, assist)
                        servo.assert_not_called()
                    else:
                        command, info = pickup.update(measured, self.snapshot, assist)
                        self.assertEqual(info['approach_component_displacement_m'], .012)
                        self.assertEqual(info['approach_displacement_m'], 0.)
                        np.testing.assert_array_equal(command, measured)

    def test_unverified_table_holds_before_servo_and_stops_after_fresh_deadline(self):
        pickup,assist,hull,measured=self.table_pickup()
        with patch('scene1.visual_pickup.table_hull',return_value=None), \
                patch.object(pickup.motion,'_arrived') as servo:
            for _ in range(29):
                self.snapshot['stamp']+=.1
                command,info=pickup.update(measured,self.snapshot,assist)
                np.testing.assert_array_equal(command,measured)
                self.assertTrue(info['table_reference_hold'])
            for _ in range(5):pickup.update(measured,self.snapshot,assist)
            self.assertEqual(pickup.unverified_table,29)
            self.snapshot['stamp']+=.1
            with self.assertRaisesRegex(RuntimeError,'Table reference'):
                pickup.update(measured,self.snapshot,assist)
            servo.assert_not_called()

    def test_missing_rgbd_holds_without_opening_payload_then_stops(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
        pickup._stage('carry'); pickup.motion.command[15] = .98
        for _ in range(9):
            result, _ = pickup.update(measured, None, None)
            self.assertEqual(result[15], .98)
        with self.assertRaisesRegex(RuntimeError, 'lost registered RGB-D'):
            pickup.update(measured, None, None)

    def test_next_parcel_cannot_start_with_held_hand(self):
        measured = np.zeros(16); measured[15] = .5
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        with self.assertRaisesRegex(RuntimeError, 'empty and open'):
            VisualPickup(cycle, self.template, self.pad, measured, measured)

    def test_lost_contact_stops_carry_without_release(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
        pickup._stage('carry'); pickup.motion.command[15] = .98
        assist = self.transport_assist()
        with patch.object(pickup.motion, '_arrived', return_value=(False, [])):
            for _ in range(3):
                action, _ = pickup.update(measured, self.snapshot, assist)
                self.assertEqual(action[15], .98)
            with self.assertRaisesRegex(RuntimeError, 'lost held parcel'):
                pickup.update(measured, self.snapshot, assist)

    def test_observe_pad_keeps_payload_and_waits_for_full_boundary(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
        pickup._stage('observe_pad'); pickup.motion.command[15] = .98
        measured[15] = .5
        pickup.motion.geometry = dict(center=np.zeros(3), rotation=Rotation.identity(), extent=np.array([.11,.04,.1]))
        assist = self.transport_assist(confirmed=True)
        head = dict(state='look_at_target',desired_degrees=[-33.,23.],
                    command_degrees=[-33.,23.],measured_degrees=[-33.,24.5])
        np.testing.assert_allclose(pickup.camera_target(measured, assist), self.pad)
        self.snapshot['surfaces'] = [dict(position_base_m=self.pad, extent_xy_m=[.2,.2], partial=True,
            points=100, footprint_xy_m=[[.325,-.66],[.525,-.66],[.525,-.46],[.325,-.46]])]
        action, _ = pickup.update(measured, self.snapshot, assist, head)
        self.assertEqual(action[15], .98); self.assertEqual(pickup.stage, 'observe_pad')
        self.snapshot['surfaces'][0]['partial'] = False
        for _ in range(8):
            self.snapshot['stamp'] += 1
            action, _ = pickup.update(measured, self.snapshot, assist, head)
            if pickup.stage == 'carry':
                break
        self.assertEqual(action[15], .98); self.assertEqual(pickup.stage, 'carry')

    def stable_pad_reference_replay(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/pad_reference_head_motion_v51.json').read_text())
        measured = np.zeros(16)
        cycle = VisualCycle({},Arm(),np.full(16,-3.),np.full(16,3.),measured)
        pickup = VisualPickup(cycle,self.template,fixture['pad'],measured,measured)
        reference = None; changes = []
        for frame in fixture['frames']:
            if pickup._pad_reference_view_ready(frame['head'],True):
                before = reference
                reference = update_pad_reference(reference,frame['surfaces'],fixture['pad'])
                if before is not reference:changes.append(frame['step'])
        return fixture,pickup,reference,changes

    def test_actual_moving_head_footprint_cannot_replace_stable_pad_reference(self):
        fixture,_,reference,changes = self.stable_pad_reference_replay()
        self.assertEqual(changes,[fixture['expected_reference_step']])
        self.assertNotEqual(reference,fixture['old_reference'])
        old = None; old_changes = []
        for frame in fixture['frames']:
            before = old
            old = update_pad_reference(old,frame['surfaces'],fixture['pad'])
            if before is not old:old_changes.append(frame['step'])
        self.assertEqual(old_changes,[5894,5901])
        self.assertEqual(old,fixture['old_reference'])

    def test_raw_failed_pad_passes_unchanged_association_with_stable_reference(self):
        from scene1.rgbd import surface_candidates
        fixture,_,reference,_ = self.stable_pad_reference_replay()
        raw = np.load(Path(__file__).parent/'fixtures/pad_reference_head_motion_v51.npz')
        surfaces = surface_candidates(raw['rgb'],raw['depth_m'],raw['intrinsic'],
            raw['camera_to_base'],fixture['plane_z'],fixture['config'])
        snapshot = dict(surfaces=surfaces)
        before = json.dumps(reference,sort_keys=True)
        self.assertFalse(visible_pad(fixture['old_reference'],snapshot))
        self.assertTrue(visible_pad(reference,snapshot))
        self.assertEqual(json.dumps(reference,sort_keys=True),before)
        shifted = json.loads(before)
        shifted['footprint_xy_m']=(np.asarray(shifted['footprint_xy_m'])+[.04,0.]).tolist()
        self.assertFalse(visible_pad(shifted,snapshot))

    def test_pad_cache_needs_fresh_head_feedback_and_holds_arm_commands(self):
        fixture,pickup,_,_ = self.stable_pad_reference_replay()
        pickup.pad_reference_head=None
        measured=np.zeros(16);assist=SimpleNamespace(_remember=lambda snapshot:None)
        frame=next(f for f in fixture['frames'] if f['step']==fixture['expected_reference_step'])
        snapshot=dict(self.snapshot,surfaces=frame['surfaces'],stamp=frame['stamp'])
        before=pickup.motion.command.copy()
        for _ in range(4):
            action,_=pickup.update(measured,snapshot,assist,None)
            np.testing.assert_array_equal(action,before)
        self.assertIsNone(pickup.surface_reference)
        snapshot['stamp']+=1
        pickup.update(measured,snapshot,assist,frame['head'])
        for _ in range(4):pickup.update(measured,snapshot,assist,frame['head'])
        self.assertIsNone(pickup.surface_reference)
        snapshot['stamp']+=1
        action,_=pickup.update(measured,snapshot,assist,frame['head'])
        self.assertIsNotNone(pickup.surface_reference)
        np.testing.assert_array_equal(action,before)

    def test_real_occluded_pad_confirms_full_boundary_without_shrinking(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/occluded_pickup_pad.json').read_text())
        before = json.dumps(fixture['reference'], sort_keys=True)
        self.assertTrue(visible_pad(fixture['reference'], fixture))
        self.assertEqual(json.dumps(fixture['reference'], sort_keys=True), before)
        for surface in fixture['surfaces']:
            surface['footprint_xy_m'] = (np.asarray(surface['footprint_xy_m'])+[.08,0.]).tolist()
        self.assertFalse(visible_pad(fixture['reference'], fixture))

    def test_carry_handoff_requires_visual_preparation_without_releasing(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
        pickup._stage('carry'); pickup.motion.command[15] = .98
        pickup.surface = dict(position_base_m=self.pad)
        goal = np.eye(4); goal[:3,:3] = Rotation.from_euler('y', .23).as_matrix()
        pickup.motion.goal[1] = goal
        pickup.box_rotation = Rotation.from_euler('y', .23)
        pickup.box_destination = np.zeros(3)
        measured[15] = .5
        begin = []
        assist = self.transport_assist(confirmed=True)
        assist.offset = np.zeros(16)
        assist._begin = lambda *args, **kwargs: begin.append((args, kwargs))
        for _ in range(8):
            self.snapshot['stamp'] += 1
            action, _ = pickup.update(measured, self.snapshot, assist)
            self.assertEqual(action[15], .98)
            if pickup.stage == 'prepare_weighing':
                break
        self.assertEqual(pickup.stage, 'prepare_weighing')
        self.assertEqual(begin, [])

    def test_prepared_weighing_uses_separate_gain_and_restores_before_regrasp(self):
        measured = np.zeros(16)
        cycle = VisualCycle({'weighing_gain_scale': .4}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
        pickup._stage('prepare_weighing'); pickup.motion.command[15] = .98; measured[15] = .5
        pickup.motion.goal[1] = np.eye(4); pickup.box_rotation = Rotation.identity()
        pickup.box_destination = np.zeros(3); pickup.surface = dict(position_base_m=[0.,0.,-.03])
        assist = self.transport_assist(confirmed=True)
        assist._observe_box = lambda *args: dict(orientation_confirmed=True, bottom_base_m=0., tilt_degrees=0.)
        original = dict(gain=.25, max_joint_step_rad=.012, max_bottom_height_spread_m=.027)
        assist.config = original; assist.offset = np.zeros(16)
        assist._begin = lambda *args, **kwargs: setattr(assist, 'pending', dict(kind='weighing'))
        for _ in range(8):
            self.snapshot['stamp'] += 1
            pickup.update(measured, self.snapshot, assist)
            if pickup.stage == 'local_weighing':
                break
        self.assertEqual(pickup.stage, 'local_weighing')
        self.assertEqual(original['gain'], .25)
        self.assertAlmostEqual(assist.config['gain'], .1)
        self.assertEqual(assist.config['max_bottom_height_spread_m'], .027)
        assist.held_action = pickup.motion.command.copy()
        def released(*args):
            assist.pending = dict(kind='short_retreat')
            return pickup.motion.command.copy(), dict(state='weighing_release')
        assist.apply = released; assist.regrasp_complete = False
        pickup.update(measured, self.snapshot, assist)
        self.assertIs(assist.config, original)

    def test_prepare_may_match_original_level_target_but_never_exceed_it(self):
        measured=np.zeros(16);measured[15]=.5
        cycle=VisualCycle({'weighing_prepare_tilt_degrees':10.},Arm(),np.full(16,-3.),np.full(16,3.),np.zeros(16))
        pickup=VisualPickup(cycle,self.template,self.pad,np.zeros(16),np.zeros(16))
        pickup._stage('prepare_weighing');pickup.motion.command[15]=.98
        pickup.motion.goal[1]=np.eye(4);pickup.box_rotation=Rotation.identity()
        pickup.box_destination=np.zeros(3);pickup.surface=dict(position_base_m=[0.,0.,-.03])
        assist=self.transport_assist(confirmed=True)
        assist.config=dict(gain=.25,max_joint_step_rad=.012,level_target_degrees=10.)
        assist.offset=np.zeros(16);begun=[]
        assist._begin=lambda *args,**kwargs:begun.append((args,kwargs))
        assist._observe_box=lambda *args:dict(orientation_confirmed=True,bottom_base_m=0.,tilt_degrees=9.5)
        with patch.object(pickup.motion,'_arrived',return_value=(True,[])) as servo:
            for _ in range(3):
                self.snapshot['stamp']+=.1
                command,_=pickup.update(measured,self.snapshot,assist)
            self.assertEqual(pickup.stage,'local_weighing');self.assertEqual(command[15],.98)
            self.assertEqual(begun[0][0][0],'weighing')  # original controller still owns release
            self.assertAlmostEqual(servo.call_args.kwargs['angle_limit'],np.deg2rad(10.))
        pickup._stage('prepare_weighing');pickup.motion.config['weighing_prepare_tilt_degrees']=11.
        self.snapshot['stamp']+=.1
        with self.assertRaisesRegex(ValueError,'original weighing level target'):
            pickup.update(measured,self.snapshot,assist)

    def test_template_preserves_verified_grasp_height_and_insertion(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        self.template['box_to_tool_m'] = [-.03, 0., .016]
        pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
        pickup.target = dict(position_base_m=[.4,-.3,-.01], major_axis_xy=[1.,0.],
                             oriented_extent_m=[.112,.042], top_z_base_m=.02)
        pickup.plane = -.083
        goal = pickup._target_goal(measured, SimpleNamespace(_grasp_goal=lambda *args: np.array([.4,-.3,-.0315])))
        np.testing.assert_allclose(goal[:3,3], [.37,-.3,-.0155])
        np.testing.assert_allclose(pickup.target['grasp_goal_xy'], [.37,-.3])

    def test_unverified_payload_pose_cannot_start_weighing(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle, self.template, self.pad, measured, measured)
        pickup._stage('carry'); pickup.motion.command[15] = .98; measured[15] = .5
        pickup.motion.goal[1] = np.eye(4); pickup.motion.goal[1][:3,3] = [.1,.1,.1]
        assist = self.transport_assist()
        for _ in range(29):
            self.snapshot['stamp'] += 1
            action, _ = pickup.update(measured, self.snapshot, assist)
            self.assertEqual(pickup.stage, 'carry'); self.assertEqual(action[15], .98)
            np.testing.assert_array_equal(action[8:15], np.zeros(7))
        self.snapshot['stamp'] += 1
        with self.assertRaisesRegex(RuntimeError, 'pose could not be verified'):
            pickup.update(measured, self.snapshot, assist)

    def test_extra_insertion_moves_toward_center_without_changing_height_or_rotation(self):
        measured = np.zeros(16)
        cycle = VisualCycle({'pickup_extra_insertion_m': .010}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        self.template['box_to_tool_m'] = [-.03,0.,.016]
        pickup=VisualPickup(cycle,self.template,self.pad,measured,measured)
        pickup.target=dict(position_base_m=[.4,-.3,-.01],major_axis_xy=[1.,0.],
            oriented_extent_m=[.112,.042],top_z_base_m=.02)
        pickup.plane=-.083
        assist=SimpleNamespace(_grasp_goal=lambda *args:np.array([.4,-.3,-.0315]))
        goal=pickup._target_goal(measured,assist)
        np.testing.assert_allclose(goal[:3,3],[.38,-.3,-.0155])
        np.testing.assert_allclose(pickup.approach_goal[:3,3],[.37,-.3,-.0155])
        np.testing.assert_allclose(goal[:3,:3],np.eye(3))
        self.assertEqual(pickup.target['extra_insertion_m'],.01)
        # A short offset cannot cross the box center or keep accumulating.
        self.template['box_to_tool_m'][0]=-.018
        next_goal=pickup._target_goal(measured,assist)
        np.testing.assert_allclose(next_goal[:3,3],[.385,-.3,-.0155])
        np.testing.assert_allclose(pickup._target_goal(measured,assist),next_goal)

    def test_posture_preference_is_bounded_and_default_servo_is_unchanged(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        cycle.command = measured.copy(); cycle._stage('prepare_weighing')
        goal = np.eye(4)
        cycle._servo(measured, 1, goal)
        np.testing.assert_array_equal(cycle.command, measured)
        reference = np.zeros(7); reference[6] = 1.
        cycle._servo(measured, 1, goal, reference)
        self.assertGreater(cycle.command[14], 0.)
        self.assertLessEqual(abs(cycle.command[14]), .006+1e-10)
        np.testing.assert_allclose(cycle.k.matrix(cycle.command[8:15],1), goal, atol=1e-12)
        before = cycle.command.copy()
        cycle._servo(measured, 1, goal, reference, posture_gain=.3)
        self.assertLessEqual(np.max(abs(cycle.command-before)), .006+1e-10)

    def test_reserve_plan_holds_command_and_preserves_deadline_and_original_reference(self):
        measured = np.zeros(16); measured[15] = .5
        cycle = VisualCycle(dict(weighing_posture_reserve=True), Arm(),
                            np.full(16,-3.), np.full(16,3.), np.zeros(16))
        pickup = VisualPickup(cycle,self.template,self.pad,np.zeros(16),np.zeros(16))
        pickup._stage('prepare_weighing'); pickup.motion.command[15] = .98
        pickup.motion.command[12] = pickup.motion.high[12]
        pickup.motion.goal[1] = np.eye(4); pickup.box_rotation = Rotation.identity()
        pickup.box_destination = np.zeros(3); pickup.surface = dict(position_base_m=[0.,0.,-.03])
        pickup.pose_search_attempted = pickup.pose_search_done = True
        pickup.pose_frames = 3; pickup.motion.stage_steps = 200
        assist = self.transport_assist(confirmed=True)
        assist._observe_box = lambda *args: dict(orientation_confirmed=True,bottom_base_m=0.,tilt_degrees=0.)
        assist.config = dict(gain=.25,max_joint_step_rad=.012); assist.offset = np.zeros(16)
        begun=[];assist._begin=lambda *args,**kwargs:begun.append((args,kwargs))
        held=pickup.motion.command.copy();reference=held[8:15].copy();reference[4]-=.4
        plan=dict(reference=reference,predicted_position_m=.005,predicted_angle_degrees=8.,wrist_reserve_degrees=20.)
        with patch.object(pickup.motion,'_arrived',return_value=(True,[])), patch(
                'scene1.visual_pickup.regrasp_posture_reserve',return_value=plan) as planner:
            self.snapshot['stamp']+=.1
            action,info=pickup.update(measured,self.snapshot,assist)
            np.testing.assert_array_equal(action,held)
            self.assertEqual(pickup.motion.stage_steps,201)
            self.assertTrue(info['weighing_posture_reserve']['accepted'])
            for _ in range(4):
                self.snapshot['stamp']+=.1;measured[8]+=.0001
                pickup.update(measured,self.snapshot,assist)
            self.assertEqual(begun,[])
            pickup.motion.command[8:15]=reference
            for _ in range(3):
                self.snapshot['stamp']+=.1;measured[8]+=.0001
                pickup.update(measured,self.snapshot,assist)
            self.assertEqual(planner.call_count,1)
            self.assertEqual(pickup.stage,'local_weighing')
            np.testing.assert_array_equal(begun[0][1]['reference_base'],pickup.motion.command)
            self.assertEqual(pickup.motion.command[15],.98)

    def test_pose_search_holds_last_command_until_new_feedback(self):
        measured = np.zeros(16); measured[15] = .5
        cycle = VisualCycle({'weighing_pose_search': True}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle, self.template, self.pad, np.zeros(16), np.zeros(16))
        pickup._stage('prepare_weighing'); pickup.motion.command[15] = .98
        pickup.motion.command[14] = -3.
        pickup.motion.goal[1] = np.eye(4); pickup.box_rotation = Rotation.identity()
        pickup.box_destination = np.zeros(3); pickup.surface = dict(position_base_m=[0.,0.,-.03])
        assist = self.transport_assist(confirmed=True)
        assist._observe_box = lambda *args: dict(orientation_confirmed=True,bottom_base_m=0.,tilt_degrees=14.)
        def arrived(*args, **kwargs):
            pickup.motion.command[8] += .001
            return False, []
        with patch.object(pickup.motion,'_arrived',side_effect=arrived), patch(
                'scene1.visual_pickup.alternate_weighing_pose',return_value=None) as plan:
            for _ in range(40):
                self.snapshot['stamp'] += 1
                measured[8] += .0001
                before = pickup.motion.command.copy()
                action, info = pickup.update(measured,self.snapshot,assist)
                if 'weighing_pose_search' in info:
                    np.testing.assert_array_equal(action,before)
                    np.testing.assert_array_equal(plan.call_args.args[2],before)
                    self.assertFalse(info['weighing_pose_search']['found'])
                    break
            plan.assert_called_once()

    def full_orientation_search_replay(self):
        fixture = json.loads((Path(__file__).parent/'fixtures/weighing_search_full_orientation_v50.json').read_text())
        low, high = np.full(16,-3.), np.full(16,3.)
        low[8:15], high[8:15] = np.asarray(fixture['limits_right']).T
        measured = np.zeros(16)
        cycle = VisualCycle(dict(weighing_pose_search=True, weighing_prepare_tilt_degrees=10.),
            Arm(), low, high, measured)
        pickup = VisualPickup(cycle,self.template,self.pad,measured,measured)
        measured[15] = .5
        pickup._stage('prepare_weighing'); pickup.pose_frames = 3
        pickup.motion.goal[1] = np.eye(4); pickup.box_rotation = Rotation.identity()
        pickup.box_destination = np.zeros(3); pickup.surface = dict(position_base_m=[0.,0.,0.])
        return fixture, pickup, self.transport_assist(confirmed=True), measured

    def test_coupled_pose_request_only_after_search_in_configured_preparation(self):
        for enabled, searched, stage in ((False, True, 'prepare_weighing'),
                (True, False, 'prepare_weighing'), (True, True, 'prepare_weighing'),
                (True, True, 'carry')):
            fixture,pickup,assist,measured = self.full_orientation_search_replay()
            frame=fixture['frames'][0]
            pickup._stage(stage);pickup.pose_search_done=searched
            pickup.motion.config['weighing_coupled_pose_request']=enabled
            pickup.motion.stage_steps=419
            measured[:]=frame['measured'];pickup.motion.command=np.asarray(frame['command']).copy()
            pickup.surface['position_base_m'][2]=frame['seen']['bottom_base_m']-frame['gap']
            assist._observe_box=lambda *args:dict(frame['seen'])
            self.snapshot['stamp']=frame['stamp']
            with patch.object(pickup.motion,'_arrived',return_value=(False,frame['pose_errors'])) as servo:
                action,info=pickup.update(measured,self.snapshot,assist)
            self.assertEqual(servo.call_args.kwargs['coupled_pose_request'],
                             enabled and searched and stage=='prepare_weighing')
            self.assertEqual(pickup.motion.stage_steps,420)
            self.assertEqual(action[15],.98)
            self.assertEqual(pickup.stage,stage)

    def test_actual_full_orientation_failure_triggers_one_search_without_open_or_clock_reset(self):
        fixture,pickup,assist,measured = self.full_orientation_search_replay()
        triggers = []
        with patch('scene1.visual_pickup.alternate_weighing_pose',return_value=None) as planner:
            for count,frame in enumerate(fixture['frames'],1):
                measured[:] = frame['measured']
                pickup.motion.command = np.asarray(frame['command']).copy()
                before = pickup.motion.command.copy()
                pickup.surface['position_base_m'][2] = frame['seen']['bottom_base_m']-frame['gap']
                assist._observe_box = lambda *args: dict(frame['seen'])
                self.snapshot['stamp'] = frame['stamp']
                with patch.object(pickup.motion,'_arrived',return_value=(False,frame['pose_errors'])):
                    action,info = pickup.update(measured,self.snapshot,assist)
                self.assertEqual(pickup.motion.stage_steps,count)
                self.assertEqual(action[15],.98)
                self.assertEqual(pickup.stage,'prepare_weighing')
                if 'weighing_pose_search' in info:
                    triggers.append(frame['step'])
                    self.assertLessEqual(frame['seen']['tilt_degrees'],10.)
                    np.testing.assert_array_equal(action,before)
                    np.testing.assert_array_equal(planner.call_args.args[2],before)
            self.assertEqual(triggers,[fixture['first_new_trigger']])
            planner.assert_called_once()

    def test_full_orientation_search_still_requires_limit_visual_confirmation_and_safe_gap(self):
        for invalid in ('not_at_limit','unconfirmed','gap','orientation_ready'):
            with self.subTest(invalid=invalid):
                fixture,pickup,assist,measured = self.full_orientation_search_replay()
                frame = next(f for f in fixture['frames'] if f['step']==fixture['first_new_trigger'])
                pickup.motion.command = np.asarray(frame['command']).copy()
                seen = dict(frame['seen']); errors = [dict(e) for e in frame['pose_errors']]
                gap = frame['gap']
                if invalid=='not_at_limit':
                    pickup.motion.command[8:15]=(pickup.motion.low[8:15]+pickup.motion.high[8:15])/2
                elif invalid=='unconfirmed':seen['orientation_confirmed']=False
                elif invalid=='gap':gap=.04
                else:
                    seen['tilt_degrees']=9.
                    errors[0]['angle_rad']=np.deg2rad(9.)
                pickup.surface['position_base_m'][2]=seen['bottom_base_m']-gap
                assist._observe_box=lambda *args:seen
                with patch.object(pickup.motion,'_arrived',return_value=(False,errors)), patch(
                        'scene1.visual_pickup.alternate_weighing_pose') as planner:
                    def replay():
                        for _ in range(30):
                            self.snapshot['stamp']+=.1
                            action,_=pickup.update(measured,self.snapshot,assist)
                            self.assertEqual(action[15],.98)
                    if invalid=='unconfirmed':
                        with self.assertRaisesRegex(RuntimeError,'pose could not be verified'):
                            replay()
                    else:
                        replay()
                    planner.assert_not_called()

    def test_repeated_full_orientation_frame_cannot_satisfy_search_dwell(self):
        fixture,pickup,assist,measured = self.full_orientation_search_replay()
        frame = next(f for f in fixture['frames'] if f['step']==fixture['first_new_trigger'])
        pickup.motion.command=np.asarray(frame['command']).copy()
        pickup.surface['position_base_m'][2]=frame['seen']['bottom_base_m']-frame['gap']
        assist._observe_box=lambda *args:dict(frame['seen'])
        self.snapshot['stamp']=frame['stamp']
        with patch.object(pickup.motion,'_arrived',return_value=(False,frame['pose_errors'])), patch(
                'scene1.visual_pickup.alternate_weighing_pose') as planner:
            for _ in range(30):pickup.update(measured,self.snapshot,assist)
            self.assertEqual(pickup.limit_frames,1)
            planner.assert_not_called()

    def test_requested_grasp_position_is_not_replaced_by_loaded_tracking_error(self):
        target = dict(position_base_m=[.327,-.3,0.], major_axis_xy=[1.,0.],
                      oriented_extent_m=[.11,.04], top_z_base_m=.02)
        plan = grasp_plan(target, -.08, [.292,-.3,-.023])
        geometry = dict(center=np.array([.05,0.,.003]), extent=np.array([.11,.04,.1]), rotation=Rotation.identity())
        result = grasp_template(geometry, plan)
        np.testing.assert_allclose(result['box_to_tool_m'], [-.035,0.,.007])
        self.assertEqual(result['position_source'], 'first_requested_grasp_target')

    def test_repeated_rgbd_frame_cannot_start_payload_motion(self):
        measured = np.zeros(16)
        cycle = VisualCycle({}, Arm(), np.full(16,-3.), np.full(16,3.), measured)
        pickup = VisualPickup(cycle,self.template,self.pad,measured,measured)
        pickup._stage('carry'); pickup.motion.command[15] = .98; measured[15] = .5
        pickup.motion.goal[1] = np.eye(4); pickup.motion.goal[1][0,3] = .1
        assist = self.transport_assist(confirmed=True)
        for _ in range(10):
            action, info = pickup.update(measured, self.snapshot, assist)
            self.assertTrue(info['pose_observation_hold'])
            np.testing.assert_array_equal(action[8:15], np.zeros(7))

    def test_parcel_pitch_schedule_preserves_second_third_and_changes_only_fourth_rotation(self):
        for completed, pitch in [(1, 15.), (2, 15.), (3, 17.5)]:
            with self.subTest(completed=completed):
                fixture, original, assist, measured, command = self.initial_grasp_failure(True)
                original.motion.config['pickup_pitch_offset_degrees'] = 2.5
                old_goal = original._target_goal(measured, assist)
                cycle = original.motion
                cycle.completed = completed
                cycle.config['pickup_pitch_offsets_degrees'] = [0., 2.5, 2.5, 5.]
                pickup = VisualPickup(cycle, original.template, self.pad, measured, command)
                pickup.target = dict(original.target); pickup.plane = original.plane
                goal = pickup._target_goal(measured, assist)
                self.assertEqual(pickup.completed_parcels, completed)
                self.assertEqual(cycle.completed, completed)
                self.assertAlmostEqual(np.rad2deg(np.arcsin(goal[2,2])), pitch)
                np.testing.assert_array_equal(goal[:3,3], old_goal[:3,3])
                np.testing.assert_array_equal(pickup.approach_goal[:3,3], original.approach_goal[:3,3])
                if completed < 3:
                    np.testing.assert_array_equal(goal, old_goal)
                self.assertEqual(assist.wrist_config['regrasp_pitch_degrees'], 12.5)

    def test_invalid_parcel_pitch_schedule_cannot_fall_back_to_an_unchecked_pose(self):
        for offsets in ([], [0.,2.5], [0.,2.5,2.5,5.01], [0.,2.5,2.5,float('nan')],
                        [0.,2.5,2.5,-.01], [[0.,2.5,2.5,5.]]):
            with self.subTest(offsets=offsets):
                _, pickup, assist, measured, _ = self.initial_grasp_failure(True)
                pickup.completed_parcels = 3
                pickup.motion.config['pickup_pitch_offsets_degrees'] = offsets
                with self.assertRaises(ValueError):
                    pickup._target_goal(measured, assist)
