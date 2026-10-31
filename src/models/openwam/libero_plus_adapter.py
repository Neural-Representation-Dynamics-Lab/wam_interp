"""LIBERO-Plus reuses the LIBERO checkpoint and client unchanged (upstream benchmarks/libero-plus)."""

from .libero_adapter import OpenWAMLiberoAdapter


class OpenWAMLiberoPlusAdapter(OpenWAMLiberoAdapter):
    environment = "libero_plus"
