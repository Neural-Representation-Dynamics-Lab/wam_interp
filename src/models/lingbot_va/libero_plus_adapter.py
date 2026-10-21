from .libero_adapter import LingBotVALiberoAdapter


class LingBotVALiberoPlusAdapter(LingBotVALiberoAdapter):
    def postprocess_actions(self, actions, *, environment="libero_plus"):
        if environment != "libero_plus":
            raise ValueError("Expected environment='libero_plus'")
        return super().postprocess_actions(actions, environment="libero")
