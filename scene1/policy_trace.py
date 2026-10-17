"""Observe ACT's fresh chunk without changing its temporal ensemble or inputs."""


class ACTChunkTrace:
    def __init__(self, policy):
        self.chunk = None
        self.generation = 0
        self.handle = policy.model.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.chunk = output[0].detach().clone()
        self.generation += 1
        # Forward hooks must return None to leave model outputs untouched.

    def begin_inference(self):
        # A queued action must not be logged as a new model prediction.
        self.chunk = None

    def describe(self, postprocess):
        if self.chunk is None:
            return None
        result = []
        for offset in (0, 10, 20, 40, 60, 80, 99):
            if offset >= self.chunk.shape[1]:
                continue
            action = postprocess(self.chunk[:, offset].clone()).detach().cpu().numpy().reshape(-1)
            result.append(dict(offset=offset, action=action.tolist()))
        return dict(fresh_chunk=result)

    def close(self):
        self.handle.remove()
