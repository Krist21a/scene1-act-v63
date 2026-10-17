import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scene1.recovery import load_run_history, validate_act_handover_resume


class RecoveryTests(unittest.TestCase):
    def test_act_handover_retains_holding_command_and_rejects_reset_scene(self):
        measured = np.zeros(16); measured[15] = .51
        command = measured.copy(); command[15] = .98
        records = [dict(cartesian_alignment={'regrasp_contact':True},
                        measured_state=measured.tolist(),sent_action=command.tolist())]
        np.testing.assert_array_equal(validate_act_handover_resume(records,measured),command)
        for changed in (np.zeros(16), measured+np.r_[.2,np.zeros(15)]):
            with self.assertRaises(ValueError):
                validate_act_handover_resume(records,changed)
        records[0]['cartesian_alignment'] = {}
        with self.assertRaisesRegex(ValueError,'no confirmed'):
            validate_act_handover_resume(records,measured)

    def test_recovery_keeps_grasp_provenance_and_latest_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, resumed = Path(tmp) / 'pickup', Path(tmp) / 'resume'
            root.mkdir(); resumed.mkdir()
            (root / 'run.json').write_text('{}')
            (root / 'steps.jsonl').write_text(json.dumps({'state': 'close_dwell', 'sent': .1}) + '\n')
            (resumed / 'run.json').write_text(json.dumps({'resume_from': str(root)}))
            (resumed / 'steps.jsonl').write_text(json.dumps({'state': 'lift', 'sent': .7}) + '\n')
            rows = load_run_history(resumed)
            self.assertEqual(rows[0]['state'], 'close_dwell')
            self.assertEqual(rows[-1]['sent'], .7)

    def test_cycle_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'run.json').write_text(json.dumps({'resume_from': str(root)}))
            with self.assertRaisesRegex(ValueError, 'cycle'):
                load_run_history(root)
