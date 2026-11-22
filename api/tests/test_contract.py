from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.predictors import build_appraisal, get_predictor
from app.rules import CONDITION_RULE, condition_points, grade_from_issues
from app.schemas import Appraisal, FormOptions, LaptopSpecs, ModelInfo

CONTRACT_DIR = Path(__file__).resolve().parents[2] / "contract"
EXAMPLE = json.loads((CONTRACT_DIR / "example-request.json").read_text())
client = TestClient(app)


def specs(**changes) -> LaptopSpecs:
    return LaptopSpecs(**{**EXAMPLE, **changes})


@pytest.fixture(scope="module")
def predictor():
    return get_predictor()


def test_examples_match_the_schemas():
    LaptopSpecs.model_validate_json((CONTRACT_DIR / "example-request.json").read_text())
    FormOptions.model_validate_json((CONTRACT_DIR / "example-options.json").read_text())
    Appraisal.model_validate_json((CONTRACT_DIR / "example-response.json").read_text())


def test_openapi_file_is_current():
    saved = json.loads((CONTRACT_DIR / "openapi.json").read_text())
    assert saved == app.openapi(), "contract/openapi.json is stale. Run: python -m app.export_contract"


@pytest.mark.parametrize(
    ("issues", "points", "grade"),
    [
        ([], 0, "like_new"),
        (["scratches"], 1, "good"),
        (["scratches", "worn_keys"], 2, "good"),
        (["scratches", "loose_hinge"], 3, "fair"),
        (["dent_or_crack", "screen_damage"], 4, "fair"),
        (["something_broken", "loose_hinge"], 5, "worn"),
    ],
)
def test_condition_rule(issues, points, grade):
    assert condition_points(issues) == points
    assert grade_from_issues(issues) == grade


VARIANTS = {
    "example": {},
    "known gpu and screen": {"gpu": "dedicated", "screen_inches": 15.6},
    "old budget machine": {"year": 2014, "cpu": "Intel Core i3", "ram_gb": 4, "storage_gb": 128, "storage_type": "hdd"},
    "exact battery health": {"battery_health_pct": 70},
    "many issues": {"issues": ["dent_or_crack", "something_broken", "loose_hinge"], "battery_runtime": "needs_charger"},
    "like new": {"issues": [], "battery_runtime": "all_day", "year": 2025},
    "nothing optional known": {"model_name": None, "gpu": None, "screen_inches": None, "battery_runtime": "unknown"},
}


def test_options_cover_the_example(predictor):
    options = predictor.options()
    assert EXAMPLE["brand"] in options.brands
    assert EXAMPLE["laptop_type"] in options.laptop_types
    assert EXAMPLE["cpu"] in options.cpus
    assert options.year_min <= EXAMPLE["year"] <= options.year_max
    assert options.condition_rule == CONDITION_RULE


@pytest.mark.parametrize("changes", VARIANTS.values(), ids=VARIANTS.keys())
def test_slip_is_well_formed(predictor, changes):
    laptop = specs(**changes)
    slip = predictor.appraise(laptop)

    assert slip.base_value + sum(c.amount for c in slip.contributions) == slip.estimate
    assert slip.range.low <= slip.estimate <= slip.range.high
    assert slip.condition_grade == grade_from_issues(laptop.issues)
    features = [line.feature for line in slip.contributions]
    assert len(features) == len(set(features)), f"an answer has more than one line: {features}"
    for line in slip.contributions:
        assert line.feature in LaptopSpecs.model_fields, f"{line.feature!r} isn't a LaptopSpecs field"
        assert line.amount != 0, "zero lines should be dropped"
        assert "None" not in line.label, f"unknown answers shouldn't print as None: {line.label!r}"


def test_exact_battery_health_takes_priority(predictor):
    slip = predictor.appraise(specs(battery_health_pct=70))
    assert not any(line.feature == "battery_runtime" for line in slip.contributions)


def test_newest_apple_chips_can_be_priced(predictor):
    for chip in ("Apple M3", "Apple M4"):
        assert chip in predictor.options().cpus
        laptop = specs(brand="Apple", model_name="MacBook Air", laptop_type="Ultrabook", cpu=chip, year=2024)
        assert predictor.appraise(laptop).estimate > 0


def test_newer_is_not_cheaper(predictor):
    assert predictor.appraise(specs(year=2023)).estimate >= predictor.appraise(specs(year=2019)).estimate


def test_more_ram_is_not_cheaper(predictor):
    assert predictor.appraise(specs(ram_gb=32)).estimate >= predictor.appraise(specs(ram_gb=8)).estimate


def test_more_damage_is_not_more_expensive(predictor):
    worse = specs(issues=["scratches", "dent_or_crack", "loose_hinge"])
    assert predictor.appraise(worse).estimate <= predictor.appraise(specs(issues=[])).estimate


def test_build_appraisal_balances_rounding_and_keeps_sources():
    info = ModelInfo(kind="mock", name="test", version="0", mae=1000)
    slip = build_appraisal(
        specs=specs(),
        base_value=420_400.4,
        contributions=[
            ("ram_gb", "16 GB RAM", 61_499.6),
            ("cpu", "Intel Core i7", 499.0),
            ("battery_runtime", "Battery: half a day", -18_350.2, "adjustment"),
        ],
        low=400_000,
        high=500_000,
        model=info,
    )
    assert slip.base_value + sum(c.amount for c in slip.contributions) == slip.estimate
    assert all(c.amount % 1000 == 0 for c in slip.contributions)
    assert [c.feature for c in slip.contributions] == ["ram_gb", "battery_runtime"]
    assert slip.contributions[-1].source == "adjustment"
    assert slip.condition_grade == "good"


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_options_endpoint():
    response = client.get("/api/options")
    assert response.status_code == 200
    FormOptions.model_validate(response.json())


def test_appraise_endpoint():
    response = client.post("/api/appraise", json=EXAMPLE)
    assert response.status_code == 200
    slip = Appraisal.model_validate(response.json())
    assert slip.condition_grade == "good"


@pytest.mark.parametrize(
    ("changes", "why"),
    [
        ({"brand": "Nokia"}, "brand not in options"),
        ({"issues": ["scratches", "scratches"]}, "repeated issue"),
        ({"issues": ["coffee_spill"]}, "unknown issue"),
        ({"condition": "good"}, "old condition field"),
        ({"battery_runtime": None}, "battery runtime missing"),
        ({"battery_health_pct": 0}, "battery health out of range"),
        ({"gpu": "unknown"}, "not sure is null, not a string"),
    ],
)
def test_appraise_rejects_bad_requests(changes, why):
    response = client.post("/api/appraise", json={**EXAMPLE, **changes})
    assert response.status_code == 422, why
