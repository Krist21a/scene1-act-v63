"""Recover grasp provenance and the latest command across interrupted runs."""
import json
from pathlib import Path
import numpy as np
from .motion import action_vector, JOINTS


def load_run_history(directory, seen=None):
    directory = Path(directory).resolve()
    seen = set() if seen is None else seen
    if directory in seen or len(seen) >= 16:
        raise ValueError('Recovery history contains a cycle or too many runs')
    seen.add(directory)
    metadata = json.loads((directory / 'run.json').read_text())
    previous = (metadata.get('resume_from') or metadata.get('regrasp_from') or metadata.get('pickup_from')
                or metadata.get('act_handover_from'))
    records = load_run_history(previous, seen) if previous else []
    records.extend(json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines())
    if not records:
        raise ValueError('Recovery history has no control records')
    return records


def validate_act_handover_resume(records, measured):
    """Require the same still-held parcel before resuming ACT without reset."""
    if not records or not any(r.get('cartesian_alignment',{}).get('regrasp_contact') for r in records):
        raise ValueError('ACT handover source has no confirmed weighing regrasp')
    current = action_vector(measured)
    previous = action_vector(records[-1]['measured_state'])
    command = action_vector(records[-1]['sent_action'])
    if command[15] < .9 or not .20 < current[15] < .85 or current[7] > .2:
        raise ValueError('ACT handover requires the right hand still holding and the left hand open')
    if np.max(np.abs(current[JOINTS]-previous[JOINTS])) > .15:
        raise ValueError('Robot pose differs from handover source; scene may have been reset')
    return command


def load_last_arm_command(directory):
    directory = Path(directory)
    tracking = directory / 'tracking.jsonl'
    if tracking.exists():
        row = json.loads(tracking.read_text().splitlines()[-1])
        command = row['command']
    else:
        row = json.loads((directory / 'steps.jsonl').read_text().splitlines()[-1])
        command = row['sent_action']
    if len(command) != 16 or max(command[7], command[15]) > .25:
        raise ValueError('Regrasp arm-state source must contain two open grippers')
    return command
