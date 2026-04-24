"""
ARGUS TARGETING SYSTEM - Intelligence Officer Agent
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("argus.intel")


class ThreatLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class TacticalClassification(str, Enum):
    SUPPLY_LINE = "supply_line"
    DEFENSIVE_REINFORCEMENT = "defensive_reinforcement"
    FLANKING_MANEUVER = "flanking_maneuver"
    FORCE_CONCENTRATION = "force_concentration"
    WITHDRAWAL = "withdrawal"
    FEINT = "feint"
    LOGISTICS_BUILDUP = "logistics_buildup"
    RECON_SCREEN = "reconnaissance_screen"
    UNKNOWN = "unknown_intent"


class Detection(BaseModel):
    entity_id: str
    callsign: str
    entity_class: str
    latitude: float
    longitude: float
    confidence: float
    heading_deg: Optional[float] = None
    speed_kph: Optional[float] = None
    first_observed: Optional[str] = None
    observation_count: int = 1


class FormationSummary(BaseModel):
    formation_name: str
    formation_type: str
    member_callsigns: list[str]
    centroid_lat: float
    centroid_lon: float
    spread_km: float
    posture: str = "unknown"


class BaselineDeviation(BaseModel):
    location_name: str
    expected_count: int
    observed_count: int
    deviation_pct: float
    asset_types_changed: list[str]


class TacticalPayload(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    area_of_interest: str
    detections: list[Detection]
    formations: list[FormationSummary] = []
    baseline_deviations: list[BaselineDeviation] = []
    historical_context: Optional[str] = None


class TacticalBriefing(BaseModel):
    classification: TacticalClassification
    title: str
    threat_level: ThreatLevel
    situation_summary: str
    order_of_battle: str
    force_laydown: str
    intent_assessment: str
    indicators_and_warnings: list[str]
    recommended_actions: list[str]
    confidence_assessment: str
    salute_report: str
    raw_llm_response: str


INTELLIGENCE_OFFICER_SYSTEM_PROMPT = """You are ARGUS INTEL - a military intelligence analyst operating
within the Argus Targeting System. You receive structured detection data from overhead satellite
imagery (Sentinel-2, 10m resolution, processed through YOLOv8) and produce Tactical Intelligence
Briefings for operational commanders.

YOUR ANALYTICAL FRAMEWORK:
1. Order of Battle (OOB): Enumerate detected forces by type, estimated unit size, and affiliation.
2. Force Laydown: Describe the spatial disposition of forces.
3. Intent Assessment: Classify the tactical intent based on patterns.
4. SALUTE Report: Size, Activity, Location, Unit, Time, Equipment

CRITICAL CONSTRAINTS:
- You are analysing 10m GSD imagery. Individual soldiers are NOT visible.
- Confidence scores below 0.5 should be flagged as low confidence.
- Always caveat assessments with sensor limitations.
- Use NATO military terminology throughout.
- Never state certainties - use assessed, likely, possibly, cannot determine.

OUTPUT FORMAT:
Respond with a JSON object only. No markdown, no preamble.
{
  "classification": "<tactical_classification>",
  "title": "<brief descriptive title>",
  "threat_level": "<critical|high|medium|low|minimal>",
  "situation_summary": "<2-3 sentence overview>",
  "order_of_battle": "<enumeration of forces>",
  "force_laydown": "<spatial disposition analysis>",
  "intent_assessment": "<reasoning about tactical intent>",
  "indicators_and_warnings": ["<list of I&W>"],
  "recommended_actions": ["<list of recommended actions>"],
  "confidence_assessment": "<overall confidence and limiting factors>",
  "salute_report": "<formatted SALUTE>"
}"""


class IntelligenceOfficer:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("pip install anthropic")
        return self._client

    def analyze_tactical_disposition(self, payload: TacticalPayload | dict) -> TacticalBriefing:
        if isinstance(payload, dict):
            payload = TacticalPayload(**payload)

        user_message = self._build_analysis_prompt(payload)
        logger.info(f"Sending {len(payload.detections)} detections to {self.model}")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=[{
                "type": "text",
                "text": INTELLIGENCE_OFFICER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }],
            messages=[{"role": "user", "content": user_message}],
        )

        raw_text = response.content[0].text
        return self._parse_response(raw_text)

    def _build_analysis_prompt(self, payload: TacticalPayload) -> str:
        sections = [
            f"TIMESTAMP: {payload.timestamp}",
            f"AREA OF INTEREST: {payload.area_of_interest}",
            "", "=" * 60, "DETECTIONS", "=" * 60,
        ]

        for d in payload.detections:
            sections.append(
                f"  [{d.callsign}] {d.entity_class} | "
                f"({d.latitude:.6f}, {d.longitude:.6f}) | "
                f"conf={d.confidence:.2f} | obs_count={d.observation_count}"
            )

        if payload.formations:
            sections.extend(["", "=" * 60, "FORMATIONS", "=" * 60])
            for f in payload.formations:
                sections.append(
                    f"  [{f.formation_name}] {f.formation_type} | "
                    f"spread={f.spread_km:.1f}km | posture={f.posture}"
                )

        if payload.baseline_deviations:
            sections.extend(["", "=" * 60, "BASELINE DEVIATIONS", "=" * 60])
            for b in payload.baseline_deviations:
                sections.append(
                    f"  [{b.location_name}] expected={b.expected_count} "
                    f"observed={b.observed_count} deviation={b.deviation_pct:+.1f}%"
                )

        if payload.historical_context:
            sections.extend(["", "=" * 60, "HISTORICAL CONTEXT", "=" * 60, payload.historical_context])

        sections.extend([
            "", "=" * 60, "ANALYSIS TASK", "=" * 60,
            "Produce a complete Tactical Intelligence Briefing.",
        ])

        return "\n".join(sections)

    def _parse_response(self, raw_text: str) -> TacticalBriefing:
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            data = json.loads(cleaned.strip())
            return TacticalBriefing(
                classification=data.get("classification", "unknown_intent"),
                title=data.get("title", "Untitled Briefing"),
                threat_level=data.get("threat_level", "medium"),
                situation_summary=data.get("situation_summary", ""),
                order_of_battle=data.get("order_of_battle", ""),
                force_laydown=data.get("force_laydown", ""),
                intent_assessment=data.get("intent_assessment", ""),
                indicators_and_warnings=data.get("indicators_and_warnings", []),
                recommended_actions=data.get("recommended_actions", []),
                confidence_assessment=data.get("confidence_assessment", ""),
                salute_report=data.get("salute_report", ""),
                raw_llm_response=raw_text,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return TacticalBriefing(
                classification=TacticalClassification.UNKNOWN,
                title="PARSE ERROR - Manual Review Required",
                threat_level=ThreatLevel.MEDIUM,
                situation_summary=f"LLM response could not be parsed: {str(e)}",
                order_of_battle="", force_laydown="", intent_assessment="",
                indicators_and_warnings=["LLM output requires manual review"],
                recommended_actions=["Review raw LLM output"],
                confidence_assessment="Cannot assess - parse failure",
                salute_report="",
                raw_llm_response=raw_text,
            )


def analyze_tactical_disposition(json_payload: dict | str) -> dict:
    if isinstance(json_payload, str):
        json_payload = json.loads(json_payload)
    officer = IntelligenceOfficer()
    payload = TacticalPayload(**json_payload)
    briefing = officer.analyze_tactical_disposition(payload)
    return briefing.model_dump()


def build_example_payload() -> TacticalPayload:
    return TacticalPayload(
        area_of_interest="Tartus Naval Facility, Syria (35.886E, 34.896N)",
        detections=[
            Detection(entity_id="e001", callsign="DESTROYER-0001", entity_class="destroyer",
                      latitude=34.8960, longitude=35.8860, confidence=0.87, heading_deg=270,
                      speed_kph=0, observation_count=14),
            Detection(entity_id="e002", callsign="FRIGATE-0001", entity_class="frigate",
                      latitude=34.8955, longitude=35.8855, confidence=0.82, heading_deg=265,
                      speed_kph=0, observation_count=12),
            Detection(entity_id="e003", callsign="CARGO-VESSEL-0001", entity_class="cargo_vessel",
                      latitude=34.8970, longitude=35.8890, confidence=0.91, heading_deg=45,
                      speed_kph=8, observation_count=1),
        ],
        formations=[
            FormationSummary(
                formation_name="Naval Task Group Tartus", formation_type="naval_task_group",
                member_callsigns=["DESTROYER-0001", "FRIGATE-0001"],
                centroid_lat=34.8954, centroid_lon=35.8852, spread_km=0.8, posture="static"
            ),
        ],
        baseline_deviations=[
            BaselineDeviation(
                location_name="Tartus Naval Base", expected_count=5, observed_count=7,
                deviation_pct=40.0, asset_types_changed=["cargo_vessel"]
            ),
        ],
        historical_context="Tartus is Russia's sole Mediterranean naval facility.",
    )
