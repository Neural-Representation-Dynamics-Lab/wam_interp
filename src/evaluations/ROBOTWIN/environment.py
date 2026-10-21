"""Frontend for the pinned upstream task/config/reset/step APIs.

RoboTwin resolves assets relative to its checkout. The context sets cwd only
for this evaluation and restores it afterward; use one simulator per process.
"""

from contextlib import contextmanager
from pathlib import Path
import importlib
import os
import sys

from wam_interp.types import EnvironmentObservation
from ..provenance import checkout_revision
from .config import ROBOTWIN_REVISION


def observation_from_robotwin(raw, instruction, metadata=None):
    cameras = raw["observation"]
    endpose = raw.get("endpose", {})
    return EnvironmentObservation(
        cameras={"head": cameras["head_camera"]["rgb"],
                 "left_wrist": cameras["left_camera"]["rgb"],
                 "right_wrist": cameras["right_camera"]["rgb"]},
        state={"joint_position": raw["joint_action"]["vector"], **endpose},
        instruction=instruction, metadata=dict(metadata or {}),
    )


class RobotWinFrontend:
    def __init__(self, config):
        import yaml
        from envs import CONFIGS_PATH

        self.config = config
        self.root = Path(config.robotwin_root).expanduser().resolve()
        def read(path):
            with Path(path).open(encoding="utf-8") as stream:
                return yaml.safe_load(stream)
        self.args = read(self.root / "task_config" / f"{config.task_config}.yml")
        embodiments = read(Path(CONFIGS_PATH) / "_embodiment_config.yml")
        cameras = read(Path(CONFIGS_PATH) / "_camera_config.yml")
        embodiment = self.args["embodiment"]
        if len(embodiment) not in {1, 3}:
            raise ValueError("RoboTwin embodiment must contain one robot or two robots plus distance")
        left = embodiments[embodiment[0]]["file_path"]
        right = embodiments[embodiment[0] if len(embodiment) == 1 else embodiment[1]]["file_path"]
        if not left or not right:
            raise FileNotFoundError("RoboTwin embodiment asset paths are not configured")
        self.args.update(left_robot_file=left, right_robot_file=right,
                         left_embodiment_config=read(Path(left) / "config.yml"),
                         right_embodiment_config=read(Path(right) / "config.yml"),
                         dual_arm_embodied=len(embodiment) == 1)
        if len(embodiment) == 3:
            self.args["embodiment_dis"] = embodiment[2]
        head = cameras[self.args["camera"]["head_camera_type"]]
        self.args.update(head_camera_h=head["h"], head_camera_w=head["w"],
                         task_name=config.task_name, task_config=config.task_config,
                         policy_name="wam_interp", ckpt_setting="loaded_adapter", eval_mode=True,
                         eval_video_log=False)
        module = importlib.import_module(f"envs.{config.task_name}")
        self.env = getattr(module, config.task_name)()
        self.active = False
        self.env.suc = 0
        self.env.test_num = 0

    def _setup(self, seed, episode, expert=False):
        args = dict(self.args)
        if expert:
            args["render_freq"] = 0
        self.active = True  # Also clean up a partially initialized simulator.
        self.env.setup_demo(now_ep_num=episode, seed=int(seed), is_test=True, **args)

    def screen_seed(self, seed, episode):
        try:
            self._setup(seed, episode, expert=True)
            info = self.env.play_once()
            accepted = bool(self.env.plan_success and self.env.check_success())
            return accepted, info, None if accepted else "expert_failed"
        except Exception as exc:
            return False, None, f"{type(exc).__name__}: {exc}"
        finally:
            self.close()

    def begin(self, seed, episode, info, instruction_type):
        import numpy as np
        from generate_episode_instructions import generate_episode_descriptions

        self._setup(seed, episode)
        descriptions = generate_episode_descriptions(self.config.task_name, [info["info"]], self.config.instruction_pool_size)
        choices = descriptions[0][instruction_type]
        if not choices:
            raise ValueError(f"No {instruction_type} instructions for seed {seed}")
        # setup_demo establishes the upstream simulator RNG state; preserve its
        # instruction-selection order and save the chosen instruction verbatim.
        instruction = str(np.random.choice(choices))
        self.env.set_instruction(instruction=instruction)
        self.instruction = instruction
        return observation_from_robotwin(self.env.get_obs(), instruction,
                                         {"seed": seed, "task_name": self.config.task_name})

    @property
    def step_limit(self):
        return int(self.env.step_lim)

    @property
    def action_count(self):
        return int(self.env.take_action_cnt)

    def step(self, action, action_type):
        self.env.take_action(action, action_type=action_type)
        observation = observation_from_robotwin(self.env.get_obs(), self.instruction)
        return observation, bool(self.env.eval_success)

    def close(self):
        if self.active:
            self.active = False
            self.env.close_env()
            viewer = getattr(self.env, "viewer", None)
            if self.args.get("render_freq") and viewer is not None:
                viewer.close()


@contextmanager
def open_environment(config):
    root = Path(config.robotwin_root).expanduser().resolve()
    checkout_revision(root, ROBOTWIN_REVISION)
    for relative in ("envs/_base_task.py", "task_config", "description/utils/generate_episode_instructions.py", "assets"):
        if not (root / relative).exists():
            raise FileNotFoundError(f"Missing RoboTwin source or assets: {root / relative}")
    existing = sys.modules.get("envs")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Another RoboTwin checkout is already imported; use a separate process")
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    frontend = None
    try:
        os.chdir(root)
        sys.path[:0] = [str(root), str(root / "description/utils")]
        frontend = RobotWinFrontend(config)
        yield frontend
    finally:
        try:
            if frontend is not None:
                frontend.close()
        finally:
            sys.path[:] = previous_path
            os.chdir(previous_cwd)
