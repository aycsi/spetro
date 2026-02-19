import json
import pickle
from pathlib import Path

from .core.models import RoughBergomi, RoughHeston
from .neural.surrogate import NeuralSurrogate

_CLS = {"RoughBergomi": RoughBergomi, "RoughHeston": RoughHeston}


def sv_mdl(mdl, pth: str) -> None:
    cls_n = type(mdl).__name__
    if cls_n not in _CLS:
        raise ValueError("unsupported model")
    params = {k: getattr(mdl, k) for k in _CLS[cls_n].__init__.__code__.co_varnames[1:]}
    with open(Path(pth), "w") as f:
        json.dump({"cls": cls_n, "params": params}, f)


def ld_mdl(pth: str):
    with open(Path(pth)) as f:
        d = json.load(f)
    cls_n = d["cls"]
    if cls_n not in _CLS:
        raise ValueError("unsupported model")
    return _CLS[cls_n](**d["params"])


def sv_srg(srg: NeuralSurrogate, pth: str) -> None:
    bn = srg.backend_name
    tr = srg.is_trained
    if bn == "torch":
        st = srg.network.state_dict()
    else:
        st = srg.params
    with open(Path(pth), "wb") as f:
        pickle.dump({"bn": bn, "tr": tr, "st": st}, f)


def ld_srg(pth: str, eng) -> NeuralSurrogate:
    with open(Path(pth), "rb") as f:
        d = pickle.load(f)
    srg = NeuralSurrogate(eng)
    srg.is_trained = d["tr"]
    if not d["tr"]:
        return srg
    if d["bn"] == "torch":
        shp = d["st"]["network.0.weight"].shape
        inp_d = shp[1]
        srg._init_torch_components()
        net = srg.network_class(input_dim=inp_d)
        net.load_state_dict(d["st"])
        srg.network = net
    else:
        srg._init_jax_components()
        srg.params = d["st"]
        srg.network = srg.network_class()
    return srg
