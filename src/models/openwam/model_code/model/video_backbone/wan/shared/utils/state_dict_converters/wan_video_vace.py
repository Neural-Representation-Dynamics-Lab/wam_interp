# wam_interp: vendored from OpenWAM@3950b6d9 (Apache-2.0); import paths rewritten from `openwam.*`.
def VaceWanModelDictConverter(state_dict):
    state_dict_ = {name: state_dict[name] for name in state_dict if name.startswith("vace")}
    return state_dict_
