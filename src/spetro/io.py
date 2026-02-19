import json
from pathlib import Path

from .core.models import RoughBergomi, RoughHeston

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
