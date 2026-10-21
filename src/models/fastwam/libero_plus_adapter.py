"""FastWAM's LIBERO action semantics also apply to LIBERO-Plus variants."""

from .libero_adapter import FastWAMLiberoAdapter


class FastWAMLiberoPlusAdapter(FastWAMLiberoAdapter):
    def postprocess_actions(self, actions, *, environment="libero_plus"):
        if environment != "libero_plus":
            raise ValueError("Expected environment='libero_plus'")
        return super().postprocess_actions(actions, environment="libero")
