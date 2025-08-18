from __future__ import annotations

"""Utility for loading the shared YAML configuration.

Keeping configuration access in one place prevents drift across multiple
scripts (e.g. `extract.py`, `scan_catalogue.py`).  All helpers are kept
minimal and typed so they remain import-free for the rest of the project.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

__all__: Sequence[str] = [
    "load_config",
    "get_source_base_url",
    "get_user_agent",
    "get_philosopher_map",
]


_CONFIG_PATH: Path = Path(__file__).with_name("config.yaml")


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load and cache the YAML configuration file.

    Returns
    -------
    dict
        The parsed YAML configuration as a nested mapping. The result is
        cached for the lifetime of the process so repeated access is fast.
    """
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {_CONFIG_PATH}")

    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Convenience wrappers – avoid deep dict indexing in calling code

def get_source_base_url() -> str:
    """Return the base URL of the remote Daphnet site."""

    return str(load_config()["source"]["base_url"])


def get_user_agent() -> str:
    """Return the default User-Agent specified in the config."""

    return str(load_config()["source"]["user_agent"])


def get_philosopher_map() -> Mapping[int, str]:
    """Return the DK-chapter → philosopher mapping."""

    return load_config()["philosopher_map"]
