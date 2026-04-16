"""Python enum mirroring the frontend RobustnessTier."""

from enum import Enum


class RobustnessTier(str, Enum):
    BASELINE = "BASELINE"
    PARTIAL = "PARTIAL"
    TRADES = "TRADES"

    @classmethod
    def from_adv_loss_weight(cls, w: float, is_trades: bool = False) -> "RobustnessTier":
        if is_trades and w > 0.20:
            return cls.TRADES
        if w <= 0.05:
            return cls.BASELINE
        return cls.PARTIAL

    @classmethod
    def from_model_version(cls, version_id: str) -> "RobustnessTier":
        if "trades" in version_id.lower():
            return cls.TRADES
        if version_id == "stage1":
            return cls.BASELINE
        return cls.PARTIAL
