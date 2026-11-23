"""Audit native, capture and identity replay in the first three decisions of a live episode."""
from dataclasses import replace
import json
import os
from pathlib import Path
from time import perf_counter

from .future_adapter import FutureAdapter


class NativeFeatureAudit:
    def __init__(self, adapter, backend, output):
        self.adapter = adapter
        self.backend = backend
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.futures = FutureAdapter(adapter)
        if self.futures.backend != backend:
            raise ValueError("Audit backend differs from the loaded model")
        self.layer = self.futures.layer

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def predict_action(self, request):
        import torch
        if len(self.rows) >= 3:
            return self.adapter.predict_action(request)
        if request.compile_action_infer or request.seed is None:
            raise ValueError("Use eager inference and an explicit seed for replay")
        def predict(fn):
            torch.cuda.synchronize()
            started = perf_counter()
            prediction, features = fn()
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            actions = torch.as_tensor(prediction.actions).detach().float().cpu().clone()
            if not torch.isfinite(actions).all():
                raise ValueError("Nonfinite action")
            return prediction, actions, elapsed, features
        base, native, elapsed, _ = predict(lambda: (self.adapter.predict_action(request), None))
        _, captured, _, donor = predict(lambda: self.futures.capture(request))
        zero = replace(donor, parts=tuple(torch.zeros_like(v) for v in donor.parts))
        _, ablated, _, _ = predict(lambda: (self.futures.predict_with_future(request, zero), None))
        replay, restored, _, identity = predict(lambda: self.futures.capture(request, donor=donor))
        capture_error = float((native-captured).abs().max())
        replay_error = float((native-restored).abs().max())
        self.rows.append(dict(call=len(self.rows), layer=self.layer, occurrence=0, seed=request.seed,
                              native_seconds=elapsed, output_shape=list(native.shape),
                              conditioning_shapes=[list(v.shape) for v in donor.parts],
                              conditioning_dtype=str(donor.parts[0].dtype),
                              capture_max_abs=capture_error, identity_max_abs=replay_error,
                              same_features=all(torch.equal(a,b) for a,b in zip(donor.parts,identity.parts)),
                              ablation_max_abs=float((native-ablated).abs().max()),
                              native_metadata=base.metadata, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        data = dict(interface="FutureAdapter", backend=self.backend, gpu=torch.cuda.get_device_name(), chunks=self.rows)
        temp = self.output.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf8")
        os.replace(temp, self.output)
        if capture_error != 0 or replay_error != 0 or not self.rows[-1]["same_features"]:
            raise RuntimeError(f"Native parity failed; evidence in {self.output}")
        return replay
