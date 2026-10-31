"""Record load_state_dict results so loaders can fail on missing or unexpected checkpoint keys."""

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def record_state_dict_loads(modules: dict[str, Any]) -> Iterator[list[dict]]:
    """Record ``load_state_dict`` results on ``modules`` (name -> nn.Module) while active."""
    records: list[dict] = []
    patched = []
    for name, module in modules.items():
        if module is None:
            continue
        original = module.load_state_dict

        def load(state_dict, strict=True, assign=False, _original=original, _name=name):
            result = _original(state_dict, strict=strict, assign=assign)
            records.append({"module": _name, "strict": bool(strict),
                            "missing_keys": list(result.missing_keys),
                            "unexpected_keys": list(result.unexpected_keys)})
            return result

        module.load_state_dict = load
        patched.append(module)
    try:
        yield records
    finally:
        for module in patched:
            del module.load_state_dict


def check_checkpoint_loads(records: list[dict], model: Any, checkpoint: Any, *, allow_partial: bool = False) -> dict:
    """Raise if the checkpoint left weights unloaded; return a summary for provenance."""
    loaded = {record["module"] for record in records}
    problems = []
    if not loaded & {"mot", "video_expert"}:
        problems.append("no MoT/video weights were loaded")
    for record in records:
        if record["missing_keys"]:
            sample = ", ".join(record["missing_keys"][:5])
            problems.append(f"{record['module']}: {len(record['missing_keys'])} missing keys (e.g. {sample})")
    if getattr(model, "proprio_encoder", None) is not None and "proprio_encoder" not in loaded:
        problems.append("model has a proprio_encoder but the checkpoint provides no proprio_encoder weights")
    summary = {"modules_loaded": sorted(loaded),
               "missing_keys": sum(len(r["missing_keys"]) for r in records),
               "unexpected_keys": sum(len(r["unexpected_keys"]) for r in records),
               "unexpected_key_examples": [k for r in records for k in r["unexpected_keys"]][:10],
               "allow_partial": bool(allow_partial)}
    if problems and not allow_partial:
        raise RuntimeError(f"Checkpoint {checkpoint} did not load completely: " + "; ".join(problems)
                           + ". Pass allow_partial_checkpoint=True only if this is intended.")
    summary["problems"] = problems
    return summary
