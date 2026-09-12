"""JTAG scan-chain reconstruction API.

Local-only tool for electronics repair: reconstructs an unknown JTAG chain
from captured TDI/TDO bit streams and device BSDL files. Never talks to
real hardware — it only stores captures, infers chain structure and emits
JSON chain definitions and safe SVF verification sequences.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query, Response

from . import db, inference, svf
from .bsdl import BsdlError, parse_bsdl
from .consistency import analyze_consistency
from .models import (
    BsdlUpload,
    ConsistencyRequest,
    DeviceModelIn,
    InferRequest,
    device_from_in,
)

app = FastAPI(title="JTAG Scan Chain Reconstructor", version="1.0.0")
db.init()


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "jtag-recon"}


# ---------------------------------------------------------------- devices

@app.post("/devices/bsdl", status_code=201)
def upload_bsdl(req: BsdlUpload) -> dict:
    """Upload a BSDL file; returns the parsed device model and its id."""
    try:
        model = parse_bsdl(req.bsdl_text)
    except BsdlError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if req.name:
        model["name"] = req.name
    device_id = db.add_device(model["name"], req.bsdl_text, model)
    return {"device_id": device_id, "model": model}


@app.get("/devices")
def list_devices() -> list[dict]:
    return db.list_devices()


# ---------------------------------------------------------------- inference

@app.post("/infer", status_code=201)
def infer(req: InferRequest) -> dict:
    """Run chain inference. Raw captures and the resulting version are
    persisted; locks and unreliable segments are honored on (re)compute."""
    devices: list[inference.Device] = []
    for ref in req.devices:
        if ref.device_id is not None:
            row = db.get_device(ref.device_id)
            if not row:
                raise HTTPException(404, f"device id {ref.device_id} not found")
            dev = device_from_in(DeviceModelIn(**json.loads(row["model_json"])))
        elif ref.model is not None:
            dev = device_from_in(ref.model)
        else:
            raise HTTPException(422, "each device ref needs device_id or model")
        dev.count = ref.count
        devices.append(dev)

    unrel_map: dict[int, set[int]] = {}
    for u in req.unreliable:
        if u.capture_index >= len(req.captures):
            raise HTTPException(422, f"unreliable capture_index {u.capture_index} out of range")
        unrel_map.setdefault(u.capture_index, set()).update(range(u.start, u.end))

    candidates, checks = inference.infer(
        devices, req.captures, unrel_map, req.bit_order, req.locks, req.max_unknown
    )

    for cap in req.captures:
        db.add_capture(req.session, cap.kind, cap.instruction, cap.tdi, cap.tdo)

    result = {
        "candidates": [inference.candidate_to_dict(c) for c in candidates],
        "checks": checks,
        "devices": [
            {
                "name": d.name,
                "ir_length": d.ir_length,
                "ir_capture": d.ir_capture[::-1],  # back to BSDL order
                "opcodes": d.opcodes,
                "idcode_value": hex(d.idcode_value) if d.idcode_value is not None else None,
                "idcode_mask": hex(d.idcode_mask) if d.idcode_mask is not None else None,
                "boundary_length": d.boundary_length,
            }
            for d in devices
        ],
    }
    version_id = db.add_version(req.session, req.model_dump(), result, req.note)
    return {"version_id": version_id, **result}


# ---------------------------------------------------------------- storage / export

@app.get("/captures")
def get_captures(session: str | None = Query(None)) -> list[dict]:
    return db.list_captures(session)


@app.get("/versions")
def get_versions(session: str | None = Query(None)) -> list[dict]:
    return db.list_versions(session)


@app.get("/versions/{version_id}")
def get_version(version_id: int) -> dict:
    row = db.get_version(version_id)
    if not row:
        raise HTTPException(404, "version not found")
    return row


def _load_candidate(version_id: int, candidate: int) -> tuple[dict, dict, list]:
    row = db.get_version(version_id)
    if not row:
        raise HTTPException(404, "version not found")
    result = row["result"]
    cands = result.get("candidates", [])
    if candidate >= len(cands):
        raise HTTPException(404, f"candidate {candidate} not found ({len(cands)} available)")
    devices = {d["name"]: d for d in result.get("devices", [])}
    return row, cands[candidate], devices


@app.get("/versions/{version_id}/export")
def export_chain(version_id: int, candidate: int = 0) -> dict:
    """Export the chain definition as JSON."""
    row, cand, _ = _load_candidate(version_id, candidate)
    return {
        "format": "jtag-chain-definition",
        "format_version": 1,
        "source_version_id": version_id,
        "session": row["session"],
        "created_at": row["created_at"],
        "bit_order": row["request"].get("bit_order", "lsb_first"),
        "chain": cand["chain"],
        "score": cand["score"],
        "constraints": cand["constraints"],
        "unexplained": cand["unexplained"],
        "checks": row["result"]["checks"],
    }


@app.get("/versions/{version_id}/svf")
def export_svf(version_id: int, candidate: int = 0) -> Response:
    """Export a safe SVF verification sequence (IDCODE/BYPASS/SAMPLE only)."""
    _, cand, devices = _load_candidate(version_id, candidate)
    chain: list[svf.SvfDevice] = []
    for slot in cand["chain"]:
        model = devices.get(slot["device"] or "")
        if model:
            chain.append(svf.SvfDevice(
                name=model["name"],
                ir_length=slot["ir_length"],
                opcodes=model["opcodes"],
                capture=(model["ir_capture"][::-1] if model["ir_capture"] else None),
                idcode_value=(int(model["idcode_value"], 16)
                              if model["idcode_value"] else None),
                idcode_mask=(int(model["idcode_mask"], 16)
                             if model["idcode_mask"] else None),
                boundary_length=model["boundary_length"],
            ))
        else:
            chain.append(svf.SvfDevice(
                name="unknown", ir_length=slot["ir_length"], known=False,
            ))
    text = svf.generate_svf(chain)
    return Response(
        content=text,
        media_type="text/plain",
        headers={"Content-Disposition":
                 f'attachment; filename="chain_v{version_id}_c{candidate}.svf"'},
    )


# ---------------------------------------------------------------- repeat-sample consistency

@app.post("/consistency", status_code=201)
def run_consistency(req: ConsistencyRequest) -> dict:
    """Check labeled repeat IR/DR samples against a saved chain version.

    Samples are grouped by (kind, instruction), aligned per run on their TDI
    echo and mapped through the saved chain; per-bit stability, toggles and
    missing intervals are reported. Misaligned runs (TDO constant, register
    length mismatch, echo not found) are flagged with positions and excluded
    from the statistics. The batch is persisted; raw captures and existing
    versions are never modified.
    """
    version = db.get_version(req.version_id)
    if not version:
        raise HTTPException(404, f"version id {req.version_id} not found")
    if version["session"] != req.session:
        raise HTTPException(
            422, f"version {req.version_id} belongs to session "
                 f"{version['session']!r}, request session is {req.session!r}")
    runs = [r.model_dump() for r in req.runs]
    try:
        result = analyze_consistency(runs, version, req.candidate)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    batch_id = db.add_consistency_batch(
        req.session, req.version_id, req.candidate, runs, result, req.note)
    return {"batch_id": batch_id, **result}


@app.get("/consistency")
def list_consistency_batches(session: str | None = Query(None)) -> list[dict]:
    return db.list_consistency_batches(session)


@app.get("/consistency/{batch_id}")
def get_consistency_batch(batch_id: int) -> dict:
    row = db.get_consistency_batch(batch_id)
    if not row:
        raise HTTPException(404, "consistency batch not found")
    return row
