from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "stage_order": ["苗期", "开花结果期", "果实膨大期"],
    "current_stage": 0,
    "fruit_threshold": 3,
}


class ConfigManager:
    """Load and persist JSON configuration files."""

    def __init__(self, file_name: str = "config.json") -> None:
        self.root_path = Path(__file__).resolve().parents[1]
        self.config_dir = self.root_path / "configs"
        self.config_path = self.config_dir / file_name
        self.config: Dict[str, Any] = {}
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self) -> Dict[str, Any]:
        """Load configuration from disk, falling back to defaults."""
        if not self.config_path.exists():
            self.config = DEFAULT_CONFIG.copy()
            self.save()
            return self.config

        try:
            with self.config_path.open("r", encoding="utf-8") as handle:
                self.config = json.load(handle)
            return self.config
        except (OSError, json.JSONDecodeError) as exc:
            print(f"加载配置失败: {exc}")
            self.config = DEFAULT_CONFIG.copy()
            return self.config

    def save(self) -> bool:
        """Write configuration to disk."""
        try:
            with self.config_path.open("w", encoding="utf-8") as handle:
                json.dump(self.config, handle, indent=4, ensure_ascii=False)
            return True
        except OSError as exc:
            print(f"保存配置失败: {exc}")
            return False

    def update_param(self, key: str, value: Any) -> bool:
        """Update a top-level config parameter and persist it."""
        self.config[key] = value
        return self.save()

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return a config value."""
        return self.config.get(key, default)

    def reset(self) -> Dict[str, Any]:
        """Restore default configuration."""
        self.config = DEFAULT_CONFIG.copy()
        self.save()
        print("--- 配置文件已重置为默认设置 ---")
        return self.config
