"""
ARGUS TARGETING SYSTEM - FastAPI Application
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.api.intelligence_officer import (
    IntelligenceOfficer,
    TacticalPayload,
    TacticalBriefing,
    Detection,
    FormationSummary,
    BaselineDeviation,
    analyze_tactical_disposition,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("argus.api")

app = FastAPI(
    title="ARGUS Targeting System",
    description="Open-source Maven-Lite intelligence platform for satellite imagery analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestionRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    time_start: str = "2025-01-01"
    time_end: str = "2025-06-01"
    max_cloud_cover: float = 15.0
    upscale_factor: int = 4


class IngestionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    chip_count: Optional[int] = None


class DetectionSubmission(BaseModel):
    chip_id: str
    scene_id: str
    detections: list[dict]


class AnalysisRequest(BaseModel):
    area_of_interest: str
    detection_ids: Optional[list[str]] = None
    include_formations: bool = True
    include_baselines: bool = True
    historical_context: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: dict


@app.get("/")
async def root():
    return {
        "system": "ARGUS Targeting System",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        components={
            "api": "ok",
            "database": "ok" if os.getenv("DATABASE_URL") else "not_configured",
            "anthropic": "ok" if os.getenv("ANTHROPIC_API_KEY") else "not_configured",
        }
    )


@app.post("/api/v1/ingest", response_model=IngestionResponse)
async def ingest_imagery(request: IngestionRequest, background_tasks: BackgroundTasks):
    job_id = f"ingest-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    return IngestionResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion queued for bbox={request.bbox}",
    )


@app.post("/api/v1/analyze")
async def analyze_situation(request: AnalysisRequest):
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")

    from src.api.intelligence_officer import build_example_payload
    payload = build_example_payload()
    payload.area_of_interest = request.area_of_interest

    officer = IntelligenceOfficer()
    briefing = officer.analyze_tactical_disposition(payload)
    return briefing.model_dump()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
