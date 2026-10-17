#!/usr/bin/env python3
"""Read-only audit of visual completion logs; feedback is not a task verdict."""
import argparse
import json
from collections import Counter
from pathlib import Path


def analyze(directory):
    meta = json.loads((directory/'run.json').read_text())
    rows = [json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
    visual = [r for r in rows if r['cartesian_alignment'].get('strategy') == 'visual_motion']
    result = dict(run=directory.name, task=meta['task'],
                  continuation_from=meta.get('visual_cycle_from') or meta.get('visual_next_from') or meta.get('visual_weigh_from'),
                  summary=json.loads((directory/'summary.json').read_text()),
                  visual_steps=len(visual),
                  all_visual_steps_freeze_policy=all(r['policy_paused_for_alignment'] for r in visual),
                  stage_steps=dict(Counter(r['cartesian_alignment'].get('stage') for r in visual)),
                  transitions=[], gripper_events=[], whole_run_gripper_events=[], visual_confirmations=[])
    result['configured_target_parcels'] = meta['config'].get('visual_cycle', {}).get('max_parcels', 1)
    result['requested_target_parcels'] = min(meta.get('stop_after_parcels') or 4, result['configured_target_parcels'])
    result['control_elapsed_s'] = (rows[-1]['send_monotonic_s']-rows[0]['send_monotonic_s']) if rows else 0.
    result['policy_reentries'] = []
    for index, row in enumerate(rows):
        if row['cartesian_alignment'].get('state') != 'visual_cycle_complete':
            continue
        following = next((r for r in rows[index+1:] if not r['policy_paused_for_alignment']), None)
        if following is None:
            continue
        result['policy_reentries'].append(dict(
            completed_at_step=row['step'], first_inference_step=following['step'],
            prior_policy_epoch=row['policy_epoch'], new_policy_epoch=following['policy_epoch'],
            epoch_inference_index=following['epoch_inference_index'],
            act_view_ready=following['head_camera'].get('act_view_ready'),
            measured_grippers_before_inference=[following['observation_state'][7], following['observation_state'][15]]))
    grip_state = {'left': None, 'right': None}
    for r in rows:
        for side, index in (('left', 7), ('right', 15)):
            value = r['sent_action'][index]
            state = 'open' if value <= .15 else 'close' if value >= .9 else None
            if state is not None:
                if grip_state[side] is not None and state != grip_state[side]:
                    result['whole_run_gripper_events'].append(dict(step=r['step'], side=side,
                        command=state, state=r['cartesian_alignment'].get('state')))
                grip_state[side] = state
    previous = None
    for r in visual:
        info = r['cartesian_alignment']
        if info.get('transition_to'):
            result['transitions'].append(dict(step=r['step'], source=info['stage'],
                                             target=info['transition_to']))
        if info.get('bin_visual_confirmed'):
            result['visual_confirmations'].append(dict(step=r['step'], points=info['new_bin_points']))
        if previous is not None and r['step'] == previous['step']+1:
            for side, index in (('left', 7), ('right', 15)):
                before, after = previous['sent_action'][index], r['sent_action'][index]
                if (before >= .9 and after <= .15) or (before <= .15 and after >= .9):
                    result['gripper_events'].append(dict(step=r['step'], side=side,
                        command='open' if after <= .15 else 'close', stage=info['stage'],
                        feedback_before_send=[r['observation_state'][7],r['observation_state'][15]],
                        receiver_contact_frames=info.get('receiver_contact_frames'),
                        bin_footprint_clear=info.get('bin_footprint_clear')))
        previous = r
    completed = result['summary'].get('visual_confirmed_parcels') or 0
    expected = [('left', 'close', 'left_insert'), ('right', 'open', 'close_left'),
                ('left', 'open', 'bin_transport')]*completed
    observed = [(e['side'], e['command'], e['stage']) for e in result['gripper_events']]
    result['expected_gripper_sequence_matches'] = bool(completed and observed == expected)
    whole_expected = [('right', 'close'), ('right', 'open'), ('right', 'close'),
                      ('left', 'close'), ('right', 'open'), ('left', 'open')]*completed
    result['whole_run_gripper_sequence_matches'] = bool(completed and
        [(e['side'], e['command']) for e in result['whole_run_gripper_events']] == whole_expected)
    result['expected_policy_reentries_match'] = bool(
        len(result['policy_reentries']) == result['requested_target_parcels']-1 and
        all(e['new_policy_epoch'] == e['prior_policy_epoch']+1
            and e['epoch_inference_index'] == 1 and e['act_view_ready']
            and max(e['measured_grippers_before_inference']) <= .15
            for e in result['policy_reentries']))
    if meta['config'].get('visual_cycle', {}).get('visual_after_first_parcel', False):
        first_complete = next((r['step'] for r in rows if r['cartesian_alignment'].get('state') == 'visual_cycle_complete'), None)
        after_first = [r for r in rows if first_complete is not None and r['step'] > first_complete]
        result['expected_policy_reentries_match'] = not result['policy_reentries'] and all(r['policy_paused_for_alignment'] for r in after_first)
        result['visual_pickup_steps'] = sum(r['cartesian_alignment'].get('strategy') == 'visual_pickup' for r in rows)
    result['independent_visual_sequence_complete'] = bool(
        meta['task'] == 'go_run' and not result['continuation_from']
        and result['summary']['reason'] == 'visual_cycle_target_reached'
        and completed == result['requested_target_parcels']
        and result['expected_gripper_sequence_matches']
        and result['whole_run_gripper_sequence_matches']
        and result['expected_policy_reentries_match']
        and len(result['visual_confirmations']) == completed
        and result['all_visual_steps_freeze_policy'])
    # A failed four-parcel run can still contain a fully checked two-parcel
    # prefix. Report that evidence separately without relabelling the run.
    result['completion_prefixes'] = []
    first_finished = None
    for row in rows:
        info = row['cartesian_alignment']
        if info.get('state') != 'visual_cycle_complete':
            continue
        count, step = info['completed_parcels'], row['step']
        if first_finished is None:
            first_finished = step
        sequence = [(e['side'],e['command']) for e in result['whole_run_gripper_events'] if e['step'] <= step]
        wanted = [('right','close'),('right','open'),('right','close'),('left','close'),('right','open'),('left','open')]*count
        confirmations = sum(e['step'] <= step for e in result['visual_confirmations'])
        frozen = all(r['policy_paused_for_alignment'] for r in rows if first_finished < r['step'] <= step)
        result['completion_prefixes'].append(dict(completed=count,step=step,
            elapsed_s=row['send_monotonic_s']-rows[0]['send_monotonic_s'],gripper_events=len(sequence),
            exact_gripper_sequence=sequence==wanted,confirmations=confirmations,
            policy_frozen_after_first=frozen,
            independent_prefix_verified=bool(meta['task']=='go_run' and not result['continuation_from']
                and sequence==wanted and confirmations==count
                and (frozen or not meta['config'].get('visual_cycle',{}).get('visual_after_first_parcel')))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(read_only=True, runs=[analyze(path) for path in args.runs],
                  limitation='Visual and feedback evidence does not replace the official task judge.')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for r in report['runs']:
        print(r['run'], r['summary']['reason'], 'visual steps', r['visual_steps'],
              'confirmations', len(r['visual_confirmations']), 'resume', r['continuation_from'])
