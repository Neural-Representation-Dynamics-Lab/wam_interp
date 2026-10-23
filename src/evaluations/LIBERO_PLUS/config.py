from dataclasses import dataclass
from pathlib import Path

from ..LIBERO.config import LiberoEvalConfig

LIBERO_PLUS_REVISION = "4976dc30028e805ff8094b55501d532c48fec182"
SUITE_COUNTS = {"libero_spatial": 2402, "libero_object": 2518, "libero_goal": 2591, "libero_10": 2519}


@dataclass(frozen=True)
class LiberoPlusEvalConfig(LiberoEvalConfig):
    num_trials: int = 1
    benchmark_root: Path | None = None
    categories: tuple[str, ...] = ()
    difficulty_levels: tuple[int, ...] = ()

    def validate(self):
        super().validate()
        if self.task_suite_name not in SUITE_COUNTS:
            raise ValueError(f"LIBERO-Plus supports {sorted(SUITE_COUNTS)}")
        if any(level not in range(1, 6) for level in self.difficulty_levels):
            raise ValueError("difficulty_levels must contain integers from 1 to 5")
