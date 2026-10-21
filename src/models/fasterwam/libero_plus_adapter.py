from .libero_adapter import FasterWAMLiberoAdapter


class FasterWAMLiberoPlusAdapter(FasterWAMLiberoAdapter):
    def postprocess_actions(self, actions, *, environment="libero_plus"):
        if environment != "libero_plus":
            raise ValueError("Expected environment='libero_plus'")
        return super().postprocess_actions(actions, environment="libero")
