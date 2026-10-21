from wam_interp.models.fastwam.robotwin_adapter import RobotWinJointTransforms
from .adapter import FasterWAMAdapter


class FasterWAMRobotWinAdapter(FasterWAMAdapter):
    def __init__(self, model, *, data_config, dataset_stats):
        super().__init__(model)
        self.transforms = RobotWinJointTransforms(model, data_config, dataset_stats)

    def inference_defaults(self):
        return {"condition": "one_pass_future_cache", "num_video_frames": 9}

    def environment_action_type(self, environment):
        if environment != "robotwin":
            raise ValueError("Expected environment='robotwin'")
        return "qpos"

    def prepare_observation(self, observation):
        return self.transforms.prepare(observation)

    def postprocess_actions(self, actions, *, environment):
        self.environment_action_type(environment)
        return self.transforms.decode(actions)
