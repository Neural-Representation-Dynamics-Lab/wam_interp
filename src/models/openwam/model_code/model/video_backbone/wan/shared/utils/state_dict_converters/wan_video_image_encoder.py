# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
def WanImageEncoderStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("textual."):
            continue
        name_ = "model." + name
        state_dict_[name_] = state_dict[name]
    return state_dict_
