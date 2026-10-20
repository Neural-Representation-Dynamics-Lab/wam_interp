"""Faster-WAM inference with the shared FastWAM-family LIBERO transforms."""

from wam_interp.models.fastwam.libero_adapter import FastWAMLiberoAdapter
from .adapter import FasterWAMAdapter


class FasterWAMLiberoAdapter(FasterWAMAdapter):
    def __init__(self, model, *, data_config, dataset_stats, binarize_gripper=True):
        super().__init__(model)
        self.transforms = FastWAMLiberoAdapter(
            model, data_config=data_config, dataset_stats=dataset_stats,
            binarize_gripper=binarize_gripper,
        )

    def inference_defaults(self):
        return {**self.transforms.inference_defaults(), "condition": "one_pass_future_cache"}

    def prepare_observation(self, observation):
        return self.transforms.prepare_observation(observation)

    def postprocess_actions(self, actions, *, environment):
        return self.transforms.postprocess_actions(actions, environment=environment)
