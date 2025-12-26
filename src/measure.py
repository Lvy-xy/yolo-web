from __future__ import annotations

from typing import Dict

from src.config_manager import ConfigManager


class Measure:
    """Decide the growth stage from flower/fruit counts."""

    def __init__(self, config_name: str = "config.json") -> None:
        self.cfg = ConfigManager(config_name)
        self.stage_order = self.cfg.get("stage_order") or ["苗期", "开花结果期", "果实膨大期"]
        self.current_stage = self._stage_by_index(self.cfg.get("current_stage", 0))
        self.fruit_threshold = int(self.cfg.get("fruit_threshold", 3))

    def _stage_by_index(self, index: int | str) -> str:
        try:
            stage_index = int(index)
        except (TypeError, ValueError):
            stage_index = 0
        if not self.stage_order:
            return "苗期"
        return self.stage_order[min(max(stage_index, 0), len(self.stage_order) - 1)]

    def reload(self) -> str:
        """Reload config and return the refreshed current stage."""
        self.stage_order = self.cfg.get("stage_order") or ["苗期", "开花结果期", "果实膨大期"]
        self.current_stage = self._stage_by_index(self.cfg.get("current_stage", 0))
        self.fruit_threshold = int(self.cfg.get("fruit_threshold", 3))
        return self.current_stage

    def ez(self, data_dic: Dict[str, int]) -> str:
        """Return the current growth stage based on detection counts."""
        flower_count = int(data_dic.get("flower", 0) or 0)
        fruit_count = int(data_dic.get("fruit", 0) or 0)

        detected_stage = self.current_stage
        if flower_count == 0 and fruit_count == 0:
            detected_stage = self.stage_order[0]
        elif flower_count > 0 and fruit_count <= self.fruit_threshold:
            detected_stage = self.stage_order[1]
        elif fruit_count > self.fruit_threshold:
            detected_stage = self.stage_order[2]
        elif fruit_count > 0:
            detected_stage = self.stage_order[1]

        current_idx = self.stage_order.index(self.current_stage)
        detected_idx = self.stage_order.index(detected_stage)

        if detected_idx > current_idx:
            self.current_stage = detected_stage
            self.cfg.update_param("current_stage", detected_idx)
            print(f"检测到生长关键节点！状态切换至: {self.current_stage}")
        return self.current_stage
