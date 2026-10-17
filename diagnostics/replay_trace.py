#!/usr/bin/env python3
"""Replay contiguous normalized ACT chunks with the installed native ensemble.

No robot/control objects are constructed. --verify-model also checks fresh
predictions and hook equivalence on the exact saved input tensors. Missing
epoch prefixes are reported as unverified, never filled with sampled chunks.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-model', action='store_true')
    args = p.parse_args()
    import numpy as np
    import torch
    from run_scene1 import load_config
    from scene1.policy_trace import ACTChunkTrace
    _, base, _ = load_config(args.config)
    from lerobot_patches import custom_patches  # noqa: F401
    from kuavo_deploy.utils.policy_loader import load_native_policy_bundle, inject_task_prompt
    policy, preprocess, postprocess, _ = load_native_policy_bundle(
        pretrained_path=base.inference.pretrained_path, device=torch.device(base.inference.device), strict=True)
    if policy.temporal_ensembler is None:
        raise ValueError('Saved experiment requires native temporal ensembling')
    run = json.loads((args.run / 'run.json').read_text())
    if run['original_model'] != base.inference.pretrained_path:
        raise ValueError('Replay model path differs from recorded model')
    for key, value in run.get('policy_runtime', {}).items():
        if getattr(policy.config, key, None) != value:
            raise ValueError(f'Replay policy config differs: {key}')
    files = sorted((args.run / 'policy-trace').glob('frame-*.npz'))
    if not files:
        raise ValueError('No continuous trace. Old sparse observations are insufficient for ensemble replay.')
    records = []
    epoch, previous_index, valid = None, None, False
    ensemble = policy.temporal_ensembler
    device = torch.device(base.inference.device)
    with torch.inference_mode():
        for path in files:
            with np.load(path, allow_pickle=False) as data:
                meta = json.loads(str(data['metadata_json']))
                if meta['policy_paused']:
                    continue
                if 'normalized_chunk' not in data:
                    raise ValueError(f'{path}: no fresh chunk; queued-action policies need a different replay')
                chunk = torch.from_numpy(data['normalized_chunk'].copy()).to(device)
                expected = torch.from_numpy(data['selected_normalized'].copy()).to(device)
                current_epoch, index = meta['policy_epoch'], meta['epoch_inference_index']
                if current_epoch != epoch:
                    ensemble.reset()
                    valid = index == 1
                    epoch, previous_index = current_epoch, index - 1
                if index != previous_index + 1:
                    valid = False
                previous_index = index
                result = dict(step=meta['step'], policy_epoch=epoch, ensemble_verified=False,
                              action_sent=meta['action_sent'])
                if valid:
                    replayed = ensemble.update(chunk.clone())
                    torch.testing.assert_close(replayed, expected, rtol=1e-6, atol=1e-6)
                    decoded = postprocess(replayed.clone()).cpu().numpy().reshape(-1)
                    np.testing.assert_allclose(decoded, data['act_action'], rtol=1e-6, atol=1e-6)
                    result.update(ensemble_verified=True,
                                  ensemble_max_error=float((replayed - expected).abs().max()))
                else:
                    result['unverified_reason'] = 'missing_reset_prefix_or_inference_gap'
                decoded_chunk = torch.stack([postprocess(chunk[:, i].clone())
                                             for i in range(chunk.shape[1])], dim=1).cpu().numpy()[0]
                left_close = np.flatnonzero(decoded_chunk[:, 7] >= .9)
                right_open = np.flatnonzero(decoded_chunk[:, 15] <= .25)
                result.update(fresh_right=float(decoded_chunk[0, 15]),
                              fused_right=float(data['act_action'][15]),
                              predicted_first_left_close=int(left_close[0]) if left_close.size else None,
                              predicted_first_right_open=int(right_open[0]) if right_open.size else None)
                if args.verify_model:
                    observation = {key: torch.from_numpy(data[key].copy()) for key in data.files
                                   if key.startswith('observation.')}
                    inputs = preprocess(inject_task_prompt(observation, base.inference.task_prompt))
                    # predict_action_chunk does not consume/reset the replay ensemble.
                    reference = policy.predict_action_chunk(inputs).clone()
                    hook = ACTChunkTrace(policy)
                    try:
                        with_hook = policy.predict_action_chunk(inputs)
                        hook.describe(postprocess)
                        torch.testing.assert_close(reference, with_hook, rtol=0, atol=0)
                        torch.testing.assert_close(with_hook, hook.chunk, rtol=0, atol=0)
                        torch.testing.assert_close(reference, chunk, rtol=1e-5, atol=1e-5)
                    finally:
                        hook.close()
                    result['fresh_model_and_hook_verified'] = True
                records.append(result)
    with args.output.open('x') as stream:
        json.dump(dict(read_only=True, records=records,
                       verified_frames=sum(r['ensemble_verified'] for r in records),
                       unverified_frames=sum(not r['ensemble_verified'] for r in records)), stream, indent=2)
    print(f'Saved {len(records)} inference-frame comparisons to {args.output}')
    if not any(r['ensemble_verified'] for r in records):
        raise SystemExit('No complete reset prefix was available; ensemble equivalence is unverified')


if __name__ == '__main__':
    main()
