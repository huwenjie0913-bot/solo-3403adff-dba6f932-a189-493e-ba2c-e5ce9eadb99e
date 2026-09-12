"""JTAG scan-chain reconstruction API.

Local-only tool for electronics repair: reconstructs an unknown JTAG chain
from captured TDI/TDO bit streams and device BSDL files. Never talks to
real hardware — it only stores captures, infers chain structure and emits
JSON chain definitions and safe SVF verification sequences.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query, Response

from . import db, diff as diff_mod, inference, svf
from .bsdl import BsdlError, parse_bsdl
from .consistency import analyze_consistency
from .models import (
    BsdlUpload,
    ConsistencyRequest,
    DeviceModelIn,
    DiffRequest,
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


# --------------------------------------------------- version diff / migration

def _version_or_404(version_id: int) -> dict:
    version = db.get_version(version_id)
    if not version:
        raise HTTPException(404, f"version id {version_id} not found")
    return version


def _collect_diff_samples(req: DiffRequest, source_captures: list[dict]) -> tuple[list[dict], dict]:
    """Gather the source version's historical samples to migrate.

    Sources are read-only: captures embedded in the source inference
    request, saved consistency batches of that version and raw session
    captures selected by id.
    """
    samples: list[dict] = []
    selection = {"version_captures": [], "consistency_batches": [],
                 "capture_ids": []}

    if req.include_version_captures:
        for i, cap in enumerate(source_captures):
            samples.append({
                "kind": cap["kind"], "instruction": cap.get("instruction"),
                "tdi": cap["tdi"], "tdo": cap["tdo"],
                "origin": {"source": "version_request",
                           "capture_index": i},
            })
            selection["version_captures"].append(i)

    for bid in req.consistency_batch_ids:
        batch = db.get_consistency_batch(bid)
        if not batch:
            raise HTTPException(404, f"consistency batch id {bid} not found")
        if batch["session"] != req.session:
            raise HTTPException(422, f"consistency batch {bid} belongs to "
                                    f"session {batch['session']!r}")
        if batch["version_id"] != req.source_version_id:
            raise HTTPException(
                422, f"consistency batch {bid} is based on version "
                     f"{batch['version_id']}, source is "
                     f"{req.source_version_id}")
        for r in batch["runs"]:
            samples.append({
                "kind": r["kind"], "instruction": r.get("instruction"),
                "tdi": r["tdi"], "tdo": r["tdo"],
                "origin": {"source": "consistency_batch",
                           "consistency_batch_id": bid,
                           "label": r["label"], "run_index": r["run_index"]},
            })
        selection["consistency_batches"].append(bid)

    for cid in req.capture_ids:
        cap = db.get_capture(cid)
        if not cap:
            raise HTTPException(404, f"capture id {cid} not found")
        if cap["session"] != req.session:
            raise HTTPException(422, f"capture {cid} belongs to session "
                                    f"{cap['session']!r}")
        samples.append({
            "kind": cap["kind"], "instruction": cap["instruction"],
            "tdi": cap["tdi"], "tdo": cap["tdo"],
            "origin": {"source": "capture", "capture_id": cid},
        })
        selection["capture_ids"].append(cid)
    return samples, selection


@app.post("/diff", status_code=201)
def run_diff(req: DiffRequest) -> dict:
    """Compare two saved inference versions of one session and migrate the
    source version's historical samples onto the target.

    Devices are aligned by name, IDCODE, IR length, boundary length and
    unknown-slot signatures; ambiguous alignments are retained with their
    matching basis. The report lists insertions/deletions/reorderings,
    register-length changes and IR/DR bit-index maps; each migrated sample
    is judged interpretable / interpretable with gaps / contradicting /
    not interpretable with the affected register bits, missing data and
    non-migration reasons. Raw captures and both versions stay read-only;
    the JSON report is saved as an independent diff batch.
    """
    source = _version_or_404(req.source_version_id)
    target = _version_or_404(req.target_version_id)
    for which, v, cand in (
        ("source", source, req.source_candidate),
        ("target", target, req.target_candidate),
    ):
        if v["session"] != req.session:
            raise HTTPException(
                422, f"{which} version {v['id']} belongs to session "
                     f"{v['session']!r}, request session is {req.session!r}")
        if cand >= len(v["result"].get("candidates", [])):
            raise HTTPException(
                404, f"{which} candidate {cand} not found in version {v['id']}")

    samples, selection = _collect_diff_samples(
        req, source["request"].get("captures", []))

    try:
        result = diff_mod.compare_versions(
            source, target, samples,
            source_candidate=req.source_candidate,
            target_candidate=req.target_candidate)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result["sample_migration"]["selection"] = selection
    batch_id = db.add_diff_batch(
        req.session, req.source_version_id, req.source_candidate,
        req.target_version_id, req.target_candidate, result, req.note)
    return {"batch_id": batch_id, **result}


@app.get("/diff")
def list_diff_batches(session: str | None = Query(None)) -> list[dict]:
    return db.list_diff_batches(session)


@app.get("/diff/{batch_id}")
def get_diff_batch(batch_id: int) -> dict:
    row = db.get_diff_batch(batch_id)
    if not row:
        raise HTTPException(404, "diff batch not found")
    return row
