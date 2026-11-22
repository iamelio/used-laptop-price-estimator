from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.predictors import get_predictor
from app.schemas import Appraisal, FormOptions, LaptopSpecs

app = FastAPI(
    title="Tally API",
    version="0.1.0",
    description="Prices used laptops and explains each price line by line.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

predictor = get_predictor()
router = APIRouter(prefix="/api")


def _check_against_options(specs: LaptopSpecs, options: FormOptions) -> list[str]:
    problems = []
    if specs.brand not in options.brands:
        problems.append(f"Unknown brand {specs.brand!r}.")
    if specs.laptop_type not in options.laptop_types:
        problems.append(f"Unknown laptop type {specs.laptop_type!r}.")
    if specs.cpu not in options.cpus:
        problems.append(f"Unknown CPU {specs.cpu!r}.")
    if not options.year_min <= specs.year <= options.year_max:
        problems.append(f"Year must be between {options.year_min} and {options.year_max}.")
    return problems


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "model": predictor.options().model}


@router.get("/options", response_model=FormOptions)
def options() -> FormOptions:
    return predictor.options()


@router.post("/appraise", response_model=Appraisal)
def appraise(specs: LaptopSpecs) -> Appraisal:
    problems = _check_against_options(specs, predictor.options())
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    return predictor.appraise(specs)


app.include_router(router)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
