from typing import Literal
import yaml
from box import Box


def load_cfg(path, return_type: Literal["box", "dict"] = "box"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if return_type == "box":
        return Box(cfg)
    elif return_type == "dict":
        return cfg
    else:
        raise ValueError(
            f"return type '{return_type}' is invalid, available are 'box' and 'dict'"
        )
