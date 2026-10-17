"""Bounded contiguous event recorder. Never calls a policy or changes an action."""
from collections import deque
import json
from pathlib import Path


class PolicyTraceWindow:
    def __init__(self, config, output, writer=None):
        self.enabled = config.get('enabled', False)
        self.pre = config.get('pre_steps', 8)
        self.post = config.get('post_steps', 200)
        self.max_frames = config.get('max_frames', 240)
        for value in (self.pre, self.post, self.max_frames):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Trace frame limits must be nonnegative integers')
        if self.enabled and (self.post < 1 or self.max_frames < self.pre + self.post):
            raise ValueError('Trace budget must cover the pre/post window')
        self.output = Path(output)
        self.ring = deque(maxlen=self.pre)
        self.until = -1
        self.written = set()
        self.events = []
        self.writer = writer or self._write_npz

    def trigger(self, step, name):
        if not self.enabled:
            return
        self.events.append(dict(step=step, event=name))
        self.until = max(self.until, step + self.post - 1)
        for previous in self.ring:
            self._save(*previous)
        self.ring.clear()

    def _save(self, step, payload, metadata):
        if step not in self.written and len(self.written) < self.max_frames:
            self.writer(step, payload, metadata)
            self.written.add(step)

    def _write_npz(self, step, payload, metadata):
        import numpy as np
        directory = self.output / 'policy-trace'
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / f'frame-{step:05d}.npz',
                            **payload, metadata_json=json.dumps(metadata, ensure_ascii=False))

    def capture(self, step, payload, metadata):
        if not self.enabled or len(self.written) >= self.max_frames:
            return
        if step <= self.until:
            self._save(step, payload, metadata)
        else:
            self.ring.append((step, payload, metadata))

    def wants_payload(self):
        return self.enabled and len(self.written) < self.max_frames

    def close(self):
        if self.enabled:
            (self.output / 'policy-trace.json').write_text(json.dumps(dict(
                events=self.events, saved_steps=sorted(self.written),
                max_frames=self.max_frames, budget_exhausted=len(self.written) >= self.max_frames,
                note='Receipt metadata is sampled after the observation read; it is not an exact ObsBuffer frame timestamp.'), indent=2))
