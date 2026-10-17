#!/usr/bin/env python3
"""Summarize saved action evidence using stdlib only; never connect to ROS.

Forecast counts are sampled-chunk intentions, not observed task outcomes.
"""
import argparse
import json
from pathlib import Path

JOINTS = tuple(range(7)) + tuple(range(8, 15))


def analyze(directory):
    rows = [json.loads(line) for line in (directory / 'steps.jsonl').read_text().splitlines()]
    direct = [r for r in rows if r['cartesian_alignment'].get('state') in
              ('act_direct', 'act_direct_after_first_grasp') and not r['policy_paused_for_alignment']]
    first = direct[0]['step'] if direct else None
    before_receiver = []
    for row in direct:
        if row['sent_action'][7] >= .9:
            break
        before_receiver.append(row)
    trace = [r for r in before_receiver if (r.get('act_prediction') or {}).get('fresh_chunk')]
    intents, release_predictions, conflicts = [], [], []
    for r in trace:
        chunk = r['act_prediction']['fresh_chunk']
        left = max(a['action'][7] for a in chunk)
        if left >= .9:
            intents.append(r['step'])
        if left < .2 and min(a['action'][15] for a in chunk) < .25:
            release_predictions.append(r['step'])
        # act_action on a refresh step is already replaced. Use the separately
        # recorded old proposal when available, or legacy phase_refresh values.
        refresh = r['act_prediction'].get('phase_refresh', {})
        old = (r.get('act_before_refresh') or r['act_action'])[15]
        if refresh.get('refreshed'):
            old = refresh['ensembled_right']
        new = chunk[0]['action'][15]
        if new >= .95 and old <= .85 and new - old >= .2:
            conflicts.append(dict(step=r['step'], fresh_right=new, fused_right=old,
                                  sent_right=r['sent_action'][15],
                                  sampled_left_max=left))
    previous = {r['step'] + 1: r for r in rows}
    first_row = direct[0] if direct else None
    result = dict(run=directory.name, steps=len(rows), first_direct_step=first,
                  summary=json.loads((directory / 'summary.json').read_text()),
                  pre_receiver_trace_frames=len(trace),
                  sampled_receiver_intent_steps=intents,
                  sampled_open_without_receiver_steps=release_predictions,
                  fresh_close_fused_open=conflicts)
    if first_row and first in previous:
        result['first_direct_max_joint_delta_rad'] = max(abs(first_row['sent_action'][j] -
            previous[first]['sent_action'][j]) for j in JOINTS)
        result['first_direct_head'] = first_row['head_camera']
        result['first_direct_observation'] = first_row['observation_state']
    # The old logs lack message timestamps. A simultaneous rise across every
    # topic is evidence of missing updates, not proof of a particular crash.
    result['last_sensor_ages_s'] = rows[-1]['sensor_age_s']
    result['consecutive_identical_observation_tail'] = 1
    for r in reversed(rows[:-1]):
        if r['observation_state'] != rows[-1]['observation_state']:
            break
        result['consecutive_identical_observation_tail'] += 1
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = dict(read_only=True, results=[analyze(path) for path in args.runs],
                  limitations=['Sparse old chunks cannot reconstruct temporal fusion.',
                               'No parcel-success verdict is inferred from gripper values.'])
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    for r in report['results']:
        print(r['run'], 'takeover', r['first_direct_step'], 'traced', r['pre_receiver_trace_frames'],
              'receiver_intent', len(r['sampled_receiver_intent_steps']),
              'fresh_close_fused_open', len(r['fresh_close_fused_open']))


if __name__ == '__main__':
    main()
