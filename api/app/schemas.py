from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Currency = Literal["NGN"]
StorageType = Literal["ssd", "hdd"]
Gpu = Literal["integrated", "dedicated"]
Condition = Literal["like_new", "good", "fair", "worn"]
Issue = Literal[
    "scratches",
    "dent_or_crack",
    "screen_damage",
    "worn_keys",
    "something_broken",
    "loose_hinge",
    "repaired",
]
BatteryRuntime = Literal["all_day", "half_day", "one_to_three_hours", "needs_charger", "unknown"]
PredictorKind = Literal["mock", "trained"]
ContributionSource = Literal["model", "adjustment"]


class _Strict(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")


class LaptopSpecs(_Strict):
    brand: str = Field(min_length=1, examples=["Dell"])
    model_name: str | None = Field(
        default=None,
        max_length=80,
        description=(
            "Free text such as 'XPS 13'. The trained model reads the product line from it "
            "(ROG, ThinkPad, MacBook Air and so on)."
        ),
    )
    laptop_type: str = Field(min_length=1, examples=["Ultrabook"])
    year: int = Field(ge=2005, le=2100, description="Year the laptop came out.")
    cpu: str = Field(min_length=1, examples=["Intel Core i7"])
    ram_gb: int = Field(ge=1, le=256)
    storage_gb: int = Field(ge=16, le=16384)
    storage_type: StorageType
    gpu: Gpu | None = Field(default=None, description="Null when the seller isn't sure.")
    screen_inches: float | None = Field(default=None, ge=10, le=20, description="Null when the seller isn't sure.")
    issues: list[Issue] = Field(
        description="Every condition item the seller ticked. An empty list means 'None of these'."
    )
    battery_runtime: BatteryRuntime = Field(description="How long a full charge lasts with everyday use.")
    battery_health_pct: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="From the exact-health helper. When present it takes priority over battery_runtime.",
    )

    @field_validator("issues")
    @classmethod
    def _issues_unique(cls, issues: list[Issue]) -> list[Issue]:
        if len(set(issues)) != len(issues):
            raise ValueError("issues must not repeat")
        return issues


class ConditionRule(_Strict):
    points: dict[Issue, int] = Field(description="Points each issue adds.")
    min_points: dict[Condition, int] = Field(description="Fewest points needed for each grade.")

    @model_validator(mode="after")
    def _complete(self) -> ConditionRule:
        if set(self.points) != set(Issue.__args__):
            raise ValueError("points must cover every issue")
        if set(self.min_points) != set(Condition.__args__):
            raise ValueError("min_points must cover every grade")
        if self.min_points["like_new"] != 0:
            raise ValueError("like_new must start at 0 points")
        return self


class Contribution(_Strict):
    feature: str = Field(description="The LaptopSpecs field this line explains, such as 'ram_gb'.")
    label: str = Field(min_length=1, description="Human wording for the slip, such as '16 GB RAM'.")
    amount: int = Field(description="Signed naira. Positive adds to the price, negative takes away.")
    source: ContributionSource = Field(
        default="model",
        description="'model' when the model learned it; 'adjustment' when a fixed rule applied after the model "
        "set it, for answers the training data has no column for.",
    )


class PriceRange(_Strict):
    low: int = Field(gt=0)
    high: int = Field(gt=0)
    confidence: float = Field(gt=0, lt=1, description="0.8 means 8 in 10 real prices fall in this range.")


class ModelInfo(_Strict):
    kind: PredictorKind
    name: str
    version: str
    mae: int = Field(ge=0, description="Mean absolute error on the test set, in naira.")
    r2: float | None = None
    trained_on_rows: int | None = None


class Appraisal(_Strict):
    id: str
    created_at: datetime
    currency: Currency = "NGN"
    condition_grade: Condition = Field(description="Worked out from issues with the shared ConditionRule.")
    base_value: int = Field(description="Average price across the training data, before any answer counts.")
    contributions: list[Contribution]
    estimate: int = Field(gt=0)
    range: PriceRange
    model: ModelInfo

    @model_validator(mode="after")
    def _slip_adds_up(self) -> Appraisal:
        total = self.base_value + sum(c.amount for c in self.contributions)
        if total != self.estimate:
            raise ValueError(
                f"base_value + contributions = {total} but estimate = {self.estimate}. "
                "Build the response with build_appraisal() so rounding balances."
            )
        if not self.range.low <= self.estimate <= self.range.high:
            raise ValueError("estimate must sit inside range")
        return self


class FormOptions(_Strict):
    currency: Currency = "NGN"
    brands: list[str] = Field(min_length=1)
    laptop_types: list[str] = Field(min_length=1)
    cpus: list[str] = Field(min_length=1)
    ram_gb: list[int] = Field(min_length=1)
    storage_gb: list[int] = Field(min_length=1)
    screen_inches: list[float] = Field(min_length=1)
    year_min: int
    year_max: int
    condition_rule: ConditionRule
    model: ModelInfo
