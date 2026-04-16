"""Pydantic v2 request/response models."""

from datetime import datetime

from pydantic import BaseModel, Field

from .robustness_tier import RobustnessTier


class PredictionResponse(BaseModel):
    image_id: str
    prediction: str = Field(pattern="^(MEL|NON_MEL)$")
    confidence: float = Field(ge=0, le=1)
    threshold: float
    model_version: str
    robustness_tier: RobustnessTier
    inference_time_ms: float
    timestamp: datetime


class GradCAMResponse(BaseModel):
    image_id: str
    heatmap_data: list[list[float]]
    original_url: str = ""
    overlay_url: str = ""


class MetricsResponse(BaseModel):
    auc: float | None = None
    balanced_accuracy: float | None = Field(default=None, alias="balancedAccuracy")
    melanoma_sensitivity: float | None = Field(default=None, alias="melanomaSensitivity")
    non_mel_specificity: float | None = Field(default=None, alias="nonMelSpecificity")
    model_version: str = Field(alias="modelVersion")
    evaluation_type: str = Field(alias="evaluationType")

    model_config = {"populate_by_name": True}


class ExperimentResponse(BaseModel):
    version: str
    adv_loss_weight: float
    epsilon: float
    pgd_steps: int
    pgd_alpha: float
    lr: float
    epochs: int
    best_auc: float
    best_bal_acc: float
    best_sens_mel: float
    adv_bal_acc: float | None = None
    status: str


class AttackRequest(BaseModel):
    epsilon: float = Field(default=0.01, gt=0, le=0.1)
    pgd_steps: int = Field(default=10, ge=1, le=100)
    pgd_alpha: float = Field(default=0.003, gt=0, le=0.01)
    true_label: int | None = Field(default=None, ge=0, le=1)


class AttackResponse(BaseModel):
    original_prediction: str
    original_confidence: float
    adversarial_prediction: str
    adversarial_confidence: float
    epsilon: float
    pgd_steps: int
    attack_success: bool
    perturbation_l_inf: float
    true_label: int | None = None
    used_ga: bool = False


class UsageStatsResponse(BaseModel):
    total_scans: int
    mel_detections: int
    mel_rate: float
    referral_rate: float
    avg_inference_ms: float
    period_start: str
    period_end: str
    daily_counts: list[dict]


class AuditLogEntry(BaseModel):
    id: str
    action: str
    user: str
    target: str
    timestamp: str
    details: str


class GAParametersRequest(BaseModel):
    alpha: float = Field(ge=0, le=1)
    tau: float = Field(ge=0.5, le=2.0)
    theta: float = Field(ge=0.3, le=0.7)


class GAParametersResponse(BaseModel):
    success: bool
    applied_bal_acc: float
    alpha: float | None = None
    tau: float | None = None
    theta: float | None = None
    weight_cnn: float | None = None
    temperature: float | None = None
    threshold: float | None = None


class ModelRegistryEntry(BaseModel):
    id: str
    label: str
    checkpoint: str
    stage: int
    status: str
    isActive: bool
    isPending: bool
