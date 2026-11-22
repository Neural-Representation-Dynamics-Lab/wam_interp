"""Audit native, capture and identity replay in the first three decisions of a live episode."""
import json
import os
from pathlib import Path
from time import perf_counter

from .native_features import native_features


class NativeFeatureAudit:
    def __init__(self, adapter, backend, output):
        self.adapter = adapter
        self.backend = backend
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.owner = adapter.architecture.mot_driver if backend == "openwam" else adapter.model
        if backend == "fasterwam":
            layers = sorted(adapter.model.mot.condition_layer_set)
            self.layer = layers[len(layers) // 2]
        elif backend == "openwam":
            self.layer = self.owner.num_layers // 2
        elif backend == "lingbot_va":
            self.layer = len(self.owner.transformer.blocks) // 2
        elif backend == "longwam":
            self.layer = None
        else:
            raise ValueError(f"Unsupported audit backend {backend}")

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def predict_action(self, request):
        import torch
        if len(self.rows) >= 3:
            return self.adapter.predict_action(request)
        if request.compile_action_infer or request.seed is None:
            raise ValueError("Use eager inference and an explicit seed for replay")
        def predict():
            torch.cuda.synchronize()
            started = perf_counter()
            result = self.adapter.predict_action(request)
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            actions = torch.as_tensor(result.actions).detach().float().cpu().clone()
            if not torch.isfinite(actions).all():
                raise ValueError("Nonfinite action")
            return result, actions, elapsed
        base, native, elapsed = predict()
        opts = {"layer": self.layer, "stage": 0}
        with native_features(self.owner, **opts) as capture:
            _, captured, _ = predict()
        # A deliberately nonsemantic ablation checks functional connectivity at this site.
        zero = {k: tuple(torch.zeros_like(v) for v in parts) for k, parts in capture.captured.items()}
        with native_features(self.owner, donors=zero, **opts):
            _, ablated, _ = predict()
        with native_features(self.owner, donors=capture.captured, **opts) as identity:
            replay, restored, _ = predict()
        capture_error = float((native-captured).abs().max())
        replay_error = float((native-restored).abs().max())
        self.rows.append(dict(call=len(self.rows), layer=self.layer, occurrence=0, seed=request.seed,
                              native_seconds=elapsed, output_shape=list(native.shape),
                              conditioning_shapes=[list(v.shape) for v in capture.captured[0]],
                              conditioning_dtype=str(capture.captured[0][0].dtype),
                              capture_max_abs=capture_error, identity_max_abs=replay_error,
                              same_features=all(torch.equal(a,b) for a,b in zip(capture.captured[0],identity.captured[0])),
                              ablation_max_abs=float((native-ablated).abs().max()),
                              native_metadata=base.metadata, peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        data = dict(backend=self.backend, gpu=torch.cuda.get_device_name(), chunks=self.rows)
        temp = self.output.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False), encoding="utf8")
        os.replace(temp, self.output)
        if capture_error != 0 or replay_error != 0 or not self.rows[-1]["same_features"]:
            raise RuntimeError(f"Native parity failed; evidence in {self.output}")
        return replay
