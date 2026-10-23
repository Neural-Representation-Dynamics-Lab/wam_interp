"""Pinned LIBERO-Plus package identification and initial-state remapping."""

from pathlib import Path
import json
import re

from ..LIBERO.environment import get_benchmark_module
from ..provenance import checkout_revision
from .config import LIBERO_PLUS_REVISION, SUITE_COUNTS


def identify_benchmark(config):
    benchmark, get_path, _ = get_benchmark_module()
    module_path = Path(benchmark.__file__).resolve()
    root = Path(config.benchmark_root).expanduser().resolve() if config.benchmark_root else module_path.parents[3]
    if not module_path.is_relative_to(root):
        raise RuntimeError(f"Imported libero package {module_path} is outside {root}; use a separate LIBERO-Plus environment")
    checkout_revision(root, LIBERO_PLUS_REVISION)
    classification = module_path.parent / "task_classification.json"
    if not classification.is_file():
        raise RuntimeError("Imported ordinary LIBERO instead of LIBERO-Plus: classification file missing")
    records = json.loads(classification.read_text(encoding="utf-8"))
    suite = benchmark.get_benchmark_dict()[config.task_suite_name]()
    if int(suite.n_tasks) != SUITE_COUNTS[config.task_suite_name]:
        raise RuntimeError("LIBERO-Plus suite size differs from the pinned benchmark")
    return suite, records[config.task_suite_name], root, classification


def initial_state_path(task, root):
    """Mirror pinned benchmark.get_task_init_states path rules; retain trusted NumPy loading."""
    name = task.init_states_file
    folder = Path(root) / task.problem_folder
    extension = name.rsplit(".", 1)[-1]
    if "_language_" in name:
        name = name.split("_language_")[0] + "." + extension
    elif "_view_" in name:
        name = name.split("_view_")[0] + "." + extension
    else:
        name = re.sub(r"_table_\d+", "", name)
        name = re.sub(r"_tb_\d+", "", name)
        if "_light_" in name:
            name = name.split("_light_")[0] + "." + extension
        if "_add_" in task.init_states_file or "_level" in task.init_states_file:
            folder = Path(root) / "libero_newobj" / task.problem_folder
            name = task.init_states_file
    return folder / name


def initial_states_for_variant(suite, task_id):
    import torch
    _, get_path, _ = get_benchmark_module()
    task = suite.get_task(task_id)
    path = initial_state_path(task, get_path("init_states"))
    values = torch.load(path, weights_only=False)
    if "_add_" in task.init_states_file or "_level" in task.init_states_file:
        values = values.reshape(1, -1)
    return values
