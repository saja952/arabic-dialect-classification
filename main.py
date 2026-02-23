import json

cfg_path = "best_model/config.json"

with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["id2label"] = {
    "0": "E",
    "1": "G",
    "2": "J",
    "3": "Y"
}

cfg["label2id"] = {
    "E": 0,
    "G": 1,
    "J": 2,
    "Y": 3
}

with open(cfg_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

print(" Updated label mapping inside config.json")