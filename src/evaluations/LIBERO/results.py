"""Serializable result objects for LIBERO evaluation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeResult:
    task_id: int
    trial_index: int
    initial_state_index: int
    seed: int | None
    success: bool
    termination_reason: str
    policy_steps: int
    warmup_steps: int
    inference_calls: int
    total_reward: float
    inference_metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskResult:
    task_id: int
    task_name: str
    instruction: str
    episodes: list[EpisodeResult]

    @property
    def successes(self) -> int:
        return sum(int(episode.success) for episode in self.episodes)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return self.successes / len(self.episodes)


@dataclass
class LiberoEvaluationResult:
    task_suite_name: str
    model_name: str
    started_at: str
    duration_seconds: float
    config: dict[str, Any]
    tasks: list[TaskResult]
    output_path: str | None = None

    @property
    def successes(self) -> int:
        return sum(task.successes for task in self.tasks)

    @property
    def episodes(self) -> int:
        return sum(len(task.episodes) for task in self.tasks)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return self.successes / self.episodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_suite_name": self.task_suite_name,
            "model_name": self.model_name,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "successes": self.successes,
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "config": self.config,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "task_name": task.task_name,
                    "instruction": task.instruction,
                    "successes": task.successes,
                    "episodes": len(task.episodes),
                    "success_rate": task.success_rate,
                    "trials": [
                        {
                            "task_id": trial.task_id,
                            "trial_index": trial.trial_index,
                            "initial_state_index": trial.initial_state_index,
                            "seed": trial.seed,
                            "success": trial.success,
                            "termination_reason": trial.termination_reason,
                            "policy_steps": trial.policy_steps,
                            "warmup_steps": trial.warmup_steps,
                            "inference_calls": trial.inference_calls,
                            "total_reward": trial.total_reward,
                            "inference_metadata": trial.inference_metadata,
                        }
                        for trial in task.episodes
                    ],
                }
                for task in self.tasks
            ],
            "output_path": self.output_path,
        }
