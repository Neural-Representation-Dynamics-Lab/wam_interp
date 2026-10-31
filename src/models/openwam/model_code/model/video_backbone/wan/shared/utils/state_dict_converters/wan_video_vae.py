# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
def WanVideoVAEStateDictConverter(state_dict):
    state_dict_ = {}
    if "model_state" in state_dict:
        state_dict = state_dict["model_state"]
    for name in state_dict:
        state_dict_["model." + name] = state_dict[name]
    return state_dict_
