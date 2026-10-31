"""LIBERO-Plus uses LIBERO's observation and command conventions."""

from .libero_adapter import LongWAMLiberoAdapter


class LongWAMLiberoPlusAdapter(LongWAMLiberoAdapter):
    def postprocess_actions(self, actions, *, environment="libero_plus"):
        if environment != "libero_plus":
            raise ValueError("Expected environment='libero_plus'")
        return super().postprocess_actions(actions, environment="libero")
