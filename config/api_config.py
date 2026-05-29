"""
Gestionnaire de configuration et clés API ForestDL.
Sauvegarde dans un fichier JSON dans le répertoire du plugin.
"""

import os
import json

_CONFIG_FILENAME = "forestdl_config.json"

_DEFAULTS = {
    "openai_key": "",
    "gemini_key": "",
    "mistral_key": "",
    "deepseek_key": "",
    "odm_url": "http://localhost:3000",
    "odm_token": "",
    "output_dir": "",
    "output_format": "geojson",
}


def _config_path(plugin_dir: str) -> str:
    return os.path.join(plugin_dir, _CONFIG_FILENAME)


def load_config(plugin_dir: str) -> dict:
    path = _config_path(plugin_dir)
    cfg = dict(_DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update({k: v for k, v in saved.items() if k in _DEFAULTS})
        except Exception:
            pass
    return cfg


def save_config(plugin_dir: str, cfg: dict):
    path = _config_path(plugin_dir)
    data = {k: cfg.get(k, _DEFAULTS.get(k, "")) for k in _DEFAULTS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_key(plugin_dir: str, provider: str) -> str:
    cfg = load_config(plugin_dir)
    return cfg.get(f"{provider}_key", "")
