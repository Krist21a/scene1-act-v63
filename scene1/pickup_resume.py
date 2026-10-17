"""Exact observed payload context for bounded, same-scene weighing diagnostics."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .motion import action_vector, JOINTS


def write_pickup_checkpoint(output, step, ros_run_id, pickup, cycle, assist, measured):
    if pickup.stage not in ('carry', 'prepare_weighing', 'local_weighing'):
        return
    if pickup.stage == 'local_weighing' and (assist.pending is None or assist.pending['kind'] != 'weighing'):
        return
    geometry = assist.geometry[1]
    if geometry is None or pickup.motion.command[15] < .9:
        return
    context = cycle.state_dict()
    state = dict(schema=1, step=step, ros_run_id=ros_run_id, stage=pickup.stage,
        completed=cycle.completed, home=context['home'], rim=context['rim'],
        template=pickup.template, pad=np.asarray(pickup.pad).tolist(), target=pickup.target,
        plane=pickup.plane, surface=pickup.surface,
        geometry=dict(center=np.asarray(geometry['center']).tolist(),
                      extent=np.asarray(geometry['extent']).tolist(), rotation_xyzw=geometry['rotation'].as_quat().tolist()),
        goal_tcp=None if pickup.motion.goal[1] is None else pickup.motion.goal[1].tolist(),
        pose_search_done=pickup.pose_search_done or pickup.pose_search_attempted,
        posture_reserve_attempted=pickup.posture_reserve_attempted,
        posture_reserve_reference=(None if pickup.posture_reserve_reference is None else
                                   pickup.posture_reserve_reference.tolist()),
        command=pickup.motion.command.tolist(), measured=action_vector(measured).tolist())
    temporary = Path(output)/'visual-pickup-checkpoint.tmp'
    temporary.write_text(json.dumps(state))
    temporary.replace(Path(output)/'visual-pickup-checkpoint.json')


def restore_weighing_context(directory, cycle, assist, measured, ros_run_id):
    directory = Path(directory)
    saved = json.loads((directory/'visual-pickup-checkpoint.json').read_text())
    searched=False
    with (directory/'steps.jsonl').open() as stream:
        for line in stream:
            last=json.loads(line)
            searched |= 'weighing_pose_search' in last['cartesian_alignment']
    if saved.get('schema') != 1 or not ros_run_id or saved.get('ros_run_id') != ros_run_id:
        raise RuntimeError('Visual weighing checkpoint belongs to another ROS scene or schema')
    latest_stage = last['cartesian_alignment'].get('transition_to', last['cartesian_alignment']['stage'])
    if saved['step'] != last['step'] or saved['stage'] != latest_stage:
        raise RuntimeError('Visual weighing checkpoint is stale relative to the last sent action')
    if saved['stage'] not in ('carry', 'prepare_weighing', 'local_weighing'):
        raise RuntimeError('Unsupported visual weighing continuation stage')
    if not 0 < saved['completed'] < cycle.config['max_parcels']:
        raise RuntimeError('Visual weighing continuation has invalid parcel completion count')
    sent, expected, measured = map(action_vector, (saved['command'], saved['measured'], measured))
    if not np.allclose(sent, last['sent_action'], atol=1e-7, rtol=0):
        raise RuntimeError('Visual weighing command differs from the recorded send')
    if sent[15] < .9 or not .2 < measured[15] < .85 or max(sent[7], measured[7]) > .15:
        raise RuntimeError('Visual weighing continuation requires the same right-hand payload')
    low = np.minimum(sent[JOINTS], expected[JOINTS])-.15
    high = np.maximum(sent[JOINTS], expected[JOINTS])+.15
    if np.any(measured[JOINTS] < low) or np.any(measured[JOINTS] > high):
        raise RuntimeError('Visual weighing physical pose differs from its checkpoint')
    cycle.home = np.asarray(saved['home']['array'])
    cycle.completed = saved['completed']
    cycle.rim = {k: np.asarray(v['array'], dtype=v['dtype']) if isinstance(v, dict) and 'array' in v else v
                 for k, v in saved['rim'].items()}
    geometry = dict(center=np.asarray(saved['geometry']['center']), extent=np.asarray(saved['geometry']['extent']),
                    rotation=Rotation.from_quat(saved['geometry']['rotation_xyzw']))
    logged_goal=last['cartesian_alignment'].get('goal_tcp',[None,None])[1]
    stored_goal=saved.get('goal_tcp')
    if stored_goal is not None and logged_goal is not None and not np.allclose(stored_goal,logged_goal):
        raise RuntimeError('Visual weighing Cartesian goal differs from its recorded goal')
    goal=np.asarray(stored_goal if stored_goal is not None else logged_goal,float)
    if goal.shape!=(4,4) or not np.isfinite(goal).all():
        raise RuntimeError('No exact Cartesian weighing goal in the checkpoint or last record')
    destination=goal[:3,3]+Rotation.from_matrix(goal[:3,:3]).apply(geometry['center'])
    assist.closed[1] = assist.first_grasp_complete = True
    assist.geometry[1] = geometry
    assist.observe_command(sent)
    recovery = dict(stage='prepare_weighing', target=saved['target'], plane=saved['plane'],
                    surface=saved['surface'], geometry=geometry, goal=goal,
                    weighing_target_xy=destination[:2],
                    posture_reserve_attempted=bool(saved.get('posture_reserve_attempted', False)),
                    posture_reserve_reference=saved.get('posture_reserve_reference'),
                    pose_search_done=bool(saved.get('pose_search_done') or searched))
    return saved['template'], np.asarray(saved['pad']), sent, recovery
