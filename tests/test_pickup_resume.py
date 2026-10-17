import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from scene1.visual_cycle import VisualCycle
from scene1.visual_pickup import VisualPickup, restore_next_context
from scene1.pickup_resume import write_pickup_checkpoint, restore_weighing_context
from test_visual_cycle import Arm


class PickupResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        measured = np.zeros(16)
        self.cycle = VisualCycle(dict(max_parcels=2,bin_lateral_offsets_m=[-.035,.035]),
                                 Arm(), np.full(16,-3.), np.full(16,3.), measured)
        self.cycle.completed = 1
        self.cycle.rim = dict(center=np.array([.7,.3]), corners=np.array([[.5,.1],[.9,.1],[.9,.5],[.5,.5]],np.float32), rim_z=.2)
        pickup = VisualPickup(self.cycle, {}, [0.,0.,0.], measured, measured)
        pickup._stage('local_weighing'); pickup.motion.command[15] = .98
        pickup.target = dict(position_base_m=[.4,-.3,0.]); pickup.plane = 0.
        pickup.surface = dict(position_base_m=[.4,-.5,0.])
        pickup.motion.goal[1]=np.eye(4);pickup.motion.goal[1][:3,3]=[.39,-.47,.07]
        geometry = dict(center=np.array([0.,0.,.01]), extent=np.array([.11,.04,.1]), rotation=Rotation.identity())
        self.assist = SimpleNamespace(geometry=[None,geometry], pending=dict(kind='weighing'),
            closed=[False,False], observe_command=lambda command: None)
        self.measured = measured.copy(); self.measured[15] = .5
        write_pickup_checkpoint(self.directory, 4, 'scene-one', pickup, self.cycle, self.assist, self.measured)
        self.row = dict(step=4, sent_action=pickup.motion.command.tolist(), cartesian_alignment=dict(stage='local_weighing',goal_tcp=[None,pickup.motion.goal[1].tolist()]))
        self.write_row()

    def write_row(self):
        (self.directory/'steps.jsonl').write_text(json.dumps(self.row)+'\n')

    def test_same_scene_payload_reenters_visual_preparation(self):
        _, _, sent, recovery = restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-one')
        self.assertEqual(sent[15], .98)
        self.assertEqual(recovery['stage'], 'prepare_weighing')
        np.testing.assert_allclose(recovery['geometry']['center'], [0.,0.,.01])

    def test_reserve_attempt_and_active_preference_survive_exact_resume(self):
        path=self.directory/'visual-pickup-checkpoint.json'
        saved=json.loads(path.read_text())
        saved['posture_reserve_attempted']=True
        saved['posture_reserve_reference']=[0.,0.,0.,0.,1.2,-.5,0.]
        path.write_text(json.dumps(saved))
        _,_,sent,recovery=restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-one')
        pickup=VisualPickup(self.cycle,{},[.4,-.5,0.],self.measured,sent,recovery)
        self.assertTrue(pickup.posture_reserve_attempted)
        np.testing.assert_array_equal(pickup.posture_reserve_reference,saved['posture_reserve_reference'])

    def test_different_scene_or_missing_contact_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'another ROS scene'):
            restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-two')
        self.measured[15] = .98
        with self.assertRaisesRegex(RuntimeError,'same right-hand payload'):
            restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-one')

    def test_checkpoint_before_later_action_cannot_be_replayed(self):
        self.row['step'] += 1; self.write_row()
        with self.assertRaisesRegex(RuntimeError,'stale'):
            restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-one')

    def test_resume_keeps_selected_goal_and_does_not_repeat_pose_search(self):
        self.row['cartesian_alignment']['weighing_pose_search']=dict(found=True)
        self.write_row()
        _,_,sent,recovery=restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-one')
        np.testing.assert_allclose(recovery['goal'][:3,3],[.39,-.47,.07])
        np.testing.assert_allclose(recovery['weighing_target_xy'],[.39,-.47])
        pickup=VisualPickup(self.cycle,{},[.4,-.5,0.],self.measured,sent,recovery)
        self.assertTrue(pickup.pose_search_done)
        np.testing.assert_allclose(pickup.weighing_target_xy,[.39,-.47])
        pickup._stage('local_weighing')
        write_pickup_checkpoint(self.directory,5,'scene-one',pickup,self.cycle,self.assist,self.measured)
        saved=json.loads((self.directory/'visual-pickup-checkpoint.json').read_text())
        self.assertTrue(saved['pose_search_done'])

    def test_legacy_checkpoint_uses_exact_logged_goal_and_rejects_missing_goal(self):
        path=self.directory/'visual-pickup-checkpoint.json';saved=json.loads(path.read_text())
        saved.pop('goal_tcp');path.write_text(json.dumps(saved))
        _,_,_,recovery=restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-one')
        np.testing.assert_allclose(recovery['goal'],self.row['cartesian_alignment']['goal_tcp'][1])
        self.row['cartesian_alignment'].pop('goal_tcp');self.write_row()
        with self.assertRaisesRegex(RuntimeError,'No exact Cartesian'):
            restore_weighing_context(self.directory,self.cycle,self.assist,self.measured,'scene-one')

    def empty_context(self):
        self.cycle.done = True
        (self.directory/'visual-cycle-checkpoint.json').write_text(json.dumps(dict(
            ros_run_id='scene-one', controller=self.cycle.state_dict(),
            pickup_template=dict(position_source='first_requested_grasp_target'), weighing_pad=[.4,-.5,0.])))
        (self.directory/'run.json').write_text(json.dumps(dict(config=dict(visual_cycle=self.cycle.config))))
        self.row.update(sent_action=[0.]*16, measured_state=[0.]*16)
        self.write_row()

    def test_empty_next_run_preserves_saved_template_without_grasp_replay(self):
        self.empty_context()
        template, pad, sent, recovery = restore_next_context(self.directory,self.cycle,self.assist,np.zeros(16),'scene-one')
        self.assertEqual(template['position_source'], 'first_requested_grasp_target')
        self.assertIsNone(recovery)
        np.testing.assert_array_equal(sent,np.zeros(16))
        np.testing.assert_allclose(pad,[.4,-.5,0.])

    def test_empty_next_run_cannot_relabel_existing_bin_slots(self):
        self.empty_context()
        self.cycle.config = dict(self.cycle.config, bin_lateral_offsets_m=[.035,.105])
        with self.assertRaisesRegex(RuntimeError,'Completed bin slots differ'):
            restore_next_context(self.directory,self.cycle,self.assist,np.zeros(16),'scene-one')

    def test_empty_approach_after_completion_may_reselect_but_later_grasp_cannot(self):
        self.empty_context()
        checkpoint=self.directory/'visual-cycle-checkpoint.json'
        saved=json.loads(checkpoint.read_text());saved['step']=3;checkpoint.write_text(json.dumps(saved))
        self.row['cartesian_alignment']=dict(state='visual_pickup_descend',stage='descend',strategy='visual_pickup')
        self.write_row()
        restore_next_context(self.directory,self.cycle,self.assist,np.zeros(16),'scene-one')
        self.row['cartesian_alignment']['stage']='local_grasp';self.write_row()
        with self.assertRaisesRegex(RuntimeError,'later manipulation'):
            restore_next_context(self.directory,self.cycle,self.assist,np.zeros(16),'scene-one')

    def test_empty_next_run_preserves_depth_of_completed_slots(self):
        self.empty_context()
        self.cycle.config = dict(self.cycle.config, bin_depth_offsets_m=[0., -.025])
        restore_next_context(self.directory,self.cycle,self.assist,np.zeros(16),'scene-one')
        self.cycle.config['bin_depth_offsets_m'][0] = -.025
        with self.assertRaisesRegex(RuntimeError,'Completed bin slots differ'):
            restore_next_context(self.directory,self.cycle,self.assist,np.zeros(16),'scene-one')
