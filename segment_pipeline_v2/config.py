import yaml


REQUIRED_KEYS = [
    "DEVICE",
    "SAVE_DIR",
    "MODEL",
    "DATASET",
    "TRAIN",
    "EVAL",
    "OPTIMIZER",
    "SCHEDULER",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    for key in REQUIRED_KEYS:
        if key not in cfg:
            raise KeyError(f"Missing config key: {key}")
    return cfg
