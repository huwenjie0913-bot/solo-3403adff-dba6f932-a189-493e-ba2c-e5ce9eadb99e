"""Chain-version diff and bit-map migration.

Compares two *saved* inference versions (each with a chosen candidate) of
the same session:

* slots are aligned by device name, IDCODE, IR length, boundary length and
  unknown-slot capture signatures; when no unique alignment exists several
  optimal alignments are retained with the matching basis for every edge;
* device insertions, deletions, reorderings and register-length changes are
  reported together with IR/DR bit-index maps (IR, BYPASS, IDCODE,
  SAMPLE/PRELOAD);
* historical samples of the *source* version are replayed through the
  *target* version: per sample and per instruction the module decides
  whether the target chain still explains the capture, which register bits
  moved / disappeared / were never observed, and why a sample cannot be
  migrated.

Raw captures and both versions are only read. The produced JSON report is
stored as an independent diff batch by the API layer.

Chain/bit conventions are identical to ``inference.py`` /
``consistency.py``: chain position 0 is closest to TDI; TDO bit 0 is the
first bit clocked out and belongs to the device closest to TDO;
register-local bit 0 is the bit closest to TDO inside that register.
"""
from __future__ import annotations

from dataclasses import dataclass

from .consistency import _best_offset, _build_segments

MAX_ALIGNMENTS = 6        # distinct optimal alignments retained in the report
ENUM_NODE_LIMIT = 200_000
FULL_BITMAP_LIMIT = 256   # above this many bits only changed bits are listed

# ---------------------------------------------------------------- descriptors

@dataclass
class _Slot:
    position: int
    name: str | None
    ir_length: int
    ir_observed: str          # shift-out order as captured during inference
    ir_pattern: str           # shift-out order known pattern (X allowed)
    idcode: int | None        # model IDCODE
    idcode_observed: int | None
    idcode_mask: int | None
    boundary: int | None


@dataclass
class _VersionView:
    version_id: int
    candidate: int
    session: str
    bit_order: str
    chain: list[dict]
    devices: dict[str, dict]
    slots: list[_Slot]


def _id(value) -> int | None:
    if not value:
        return None
    return int(value, 16)


def _view(version: dict, candidate: int) -> _VersionView:
    """Build the internal view of one saved version/candidate.

    Raises ValueError when the candidate index is out of range.
    """
    result = version["result"]
    cands = result.get("candidates", [])
    if candidate >= len(cands):
        raise ValueError(
            f"candidate {candidate} not found in version {version['id']} "
            f"({len(cands)} available)")
    chain = cands[candidate]["chain"]
    devices = {d["name"]: d for d in result.get("devices", [])}
    slots: list[_Slot] = []
    for slot in chain:
        model = devices.get(slot.get("device") or "")
        name = slot.get("device")
        if model:
            pattern = model.get("ir_capture", "")[::-1] or "X" * slot["ir_length"]
            idcode = _id(model.get("idcode_value"))
            mask = _id(model.get("idcode_mask"))
            boundary = model.get("boundary_length")
        else:
            pattern = slot.get("ir_capture_observed") or "X" * slot["ir_length"]
            idcode = mask = boundary = None
        slots.append(_Slot(
            position=slot["position"],
            name=name,
            ir_length=slot["ir_length"],
            ir_observed=slot.get("ir_capture_observed") or "",
            ir_pattern=pattern,
            idcode=idcode,
            idcode_observed=_id(slot.get("idcode_observed")),
            idcode_mask=mask,
            boundary=boundary,
        ))
    return _VersionView(
        version_id=version["id"],
        candidate=candidate,
        session=version["session"],
        bit_order=version["request"].get("bit_order", "lsb_first"),
        chain=chain,
        devices=devices,
        slots=slots,
    )


# ---------------------------------------------------------------- alignment

@dataclass
class _Edge:
    src: int
    tgt: int
    score: int
    basis: list[str]


def _cap_compatible(pattern: str, observed: str) -> bool:
    """True when a fixed capture pattern (X = don't care) fits observed bits."""
    if len(pattern) != len(observed):
        return False
    return all(p == "X" or p == o for p, o in zip(pattern, observed))


def _edge_between(s: _Slot, t: _Slot) -> _Edge | None:
    """One possible source->target correspondence, with its evidence.

    Eligibility (any one is enough):

    * identical device name;
    * identical non-mask IDCODE value (a renamed / re-identified device);
    * two unknown slots with the same IR length (capture signature bonus);
    * an identified slot and an unknown slot whose observed capture still
      matches the device's fixed INSTRUCTION_CAPTURE pattern;
    * two identified devices without IDCODEs that share both IR length and
      boundary length (weak: typically an indistinguishable-parts case).
    """
    score = 0
    basis: list[str] = []

    named = s.name is not None and t.name is not None
    same_name = named and s.name == t.name
    if same_name:
        score += 100
        basis.append("name")

    sid, tid = s.idcode or s.idcode_observed, t.idcode or t.idcode_observed
    id_conflict = False
    if sid is not None and tid is not None:
        if sid == tid:
            score += 60
            basis.append("idcode")
        else:
            score -= 40
            id_conflict = True

    if s.ir_length == t.ir_length:
        score += 20
        basis.append("ir_length")
    else:
        score -= 10

    if s.boundary is not None and t.boundary is not None:
        if s.boundary == t.boundary:
            score += 20
            basis.append("boundary_length")
        else:
            score -= 10

    s_unk, t_unk = s.name is None, t.name is None
    if s_unk and t_unk:
        if s.ir_observed and s.ir_observed == t.ir_observed:
            score += 15
            basis.append("capture_signature")
    elif s_unk or t_unk:
        known, unk = (t, s) if s_unk else (s, t)
        if _cap_compatible(known.ir_pattern, unk.ir_observed):
            score += 10
            basis.append("ir_capture_pattern")

    eligible = (
        same_name
        or "idcode" in basis
        or (s_unk and t_unk and s.ir_length == t.ir_length)
        or ((s_unk or t_unk) and s.ir_length == t.ir_length
            and "ir_capture_pattern" in basis)
        or (named and "ir_length" in basis and "boundary_length" in basis
            and "idcode" not in basis)
    )
    if not eligible or (id_conflict and not same_name):
        return None
    return _Edge(s.position, t.position, score, basis)


def _build_edges(src: list[_Slot], tgt: list[_Slot]) -> dict[int, list[_Edge]]:
    edges: dict[int, list[_Edge]] = {}
    for s in src:
        es = [e for t in tgt if (e := _edge_between(s, t)) is not None]
        es.sort(key=lambda e: e.score, reverse=True)
        edges[s.position] = es
    return edges


def _enumerate_alignments(edges: dict[int, list[_Edge]],
                          n_src: int, n_tgt: int):
    """Enumerate maximum-weight bipartite matchings (bounded).

    Returns ``(best_weight, assignments, truncated)`` where each assignment is
    a ``{source_position: target_position}`` dict (unmatched slots omitted).
    Up to ``MAX_ALIGNMENTS`` distinct optimum assignments are retained.
    """
    order = sorted(edges, key=lambda p: (len(edges[p]), p))
    best: list[dict] = []
    best_weight = None
    truncated = False
    nodes = 0

    def bound_from(i: int, weight: int, used: frozenset) -> int:
        rest = 0
        for p in order[i:]:
            v = 0
            for e in edges[p]:
                if e.tgt not in used:
                    v = e.score
                    break
            rest += v
        return weight + rest

    def dfs(i: int, used: frozenset, weight: int, assign: dict) -> None:
        nonlocal best, best_weight, truncated, nodes
        nodes += 1
        if nodes > ENUM_NODE_LIMIT:
            truncated = True
            return
        if bound_from(i, weight, used) < (best_weight if best_weight is not None else -10**9):
            return
        if i == len(order):
            if best_weight is None or weight > best_weight:
                best_weight, best = weight, [dict(assign)]
            elif weight == best_weight and len(best) < MAX_ALIGNMENTS:
                if assign not in best:
                    best.append(dict(assign))
            return
        p = order[i]
        # option 1: source slot stays unmatched (device deleted)
        dfs(i + 1, used, weight, assign)
        # option 2: match to a target slot (best edges first)
        for e in edges[p]:
            if e.tgt in used:
                continue
            assign[p] = e.tgt
            dfs(i + 1, used | {e.tgt}, weight + e.score, assign)
            del assign[p]

    dfs(0, frozenset(), 0, {})
    if best_weight is None:
        best_weight = 0
    return best_weight, best, truncated


def _slot_ref(view: _VersionView, pos: int) -> dict:
    s = view.slots[pos]
    return {
        "position": pos,
        "device": s.name,
        "status": "identified" if s.name else "unknown",
        "ir_length": s.ir_length,
        "boundary_length": s.boundary,
        "idcode": hex(s.idcode) if s.idcode is not None else None,
        "idcode_observed": (hex(s.idcode_observed)
                            if s.idcode_observed is not None else None),
        "ir_capture_observed": s.ir_observed or None,
    }


def _idcode_pair(src: _VersionView, tgt: _VersionView,
                 sp: int, tp: int) -> dict:
    s, t = src.slots[sp], tgt.slots[tp]
    return {
        "source": hex(s.idcode) if s.idcode is not None else None,
        "target": hex(t.idcode) if t.idcode is not None else None,
        "source_mask": hex(s.idcode_mask) if s.idcode_mask is not None else None,
        "target_mask": hex(t.idcode_mask) if t.idcode_mask is not None else None,
    }


def _align(src: _VersionView, tgt: _VersionView) -> dict:
    edges = _build_edges(src.slots, tgt.slots)
    weight, alignments, truncated = _enumerate_alignments(
        edges, len(src.slots), len(tgt.slots))
    primary = alignments[0] if alignments else {}

    # partners each source position takes across the retained optima
    alt_partners: dict[int, set[int]] = {}
    for a in alignments[1:]:
        for sp, tp in a.items():
            if primary.get(sp) != tp:
                alt_partners.setdefault(sp, set()).add(tp)
    ambiguous = bool(alt_partners) or truncated

    pairs: list[dict] = []
    matched_src: set[int] = set()
    matched_tgt: set[int] = set()
    for sp in sorted(primary):
        tp = primary[sp]
        matched_src.add(sp)
        matched_tgt.add(tp)
        s, t = src.slots[sp], tgt.slots[tp]
        edge = next(e for e in edges[sp] if e.tgt == tp)
        if s.name and t.name and s.name == t.name:
            match_type = "identical"
        elif "idcode" in edge.basis:
            match_type = "renamed_by_idcode"
        elif s.name is None or t.name is None:
            match_type = "identity_unresolved"
        else:
            match_type = "weak"
        changes = []
        if s.ir_length != t.ir_length:
            changes.append("ir_length_changed")
        if s.boundary != t.boundary:
            changes.append("boundary_length_changed")
        if (s.idcode or s.idcode_observed) != (t.idcode or t.idcode_observed):
            changes.append("idcode_changed")
        if s.ir_observed and t.ir_observed and s.ir_observed != t.ir_observed:
            changes.append("ir_capture_changed")
        alternatives = []
        for atp in sorted(alt_partners.get(sp, set())):
            ae = next((e for e in edges[sp] if e.tgt == atp), None)
            alternatives.append({
                **_slot_ref(tgt, atp),
                "score": ae.score if ae else None,
                "basis": ae.basis if ae else [],
            })
        pairs.append({
            "source_position": sp,
            "target_position": tp,
            "source_device": s.name,
            "target_device": t.name,
            "match_type": match_type,
            "basis": edge.basis,
            "score": edge.score,
            "changes": changes,
            "ir_length": [s.ir_length, t.ir_length],
            "boundary_length": [s.boundary, t.boundary],
            "idcode": _idcode_pair(src, tgt, sp, tp),
            "alternatives": alternatives,
        })

    inserted = [_slot_ref(tgt, tp) for tp in range(len(tgt.slots))
                if tp not in matched_tgt]
    deleted = [_slot_ref(src, sp) for sp in range(len(src.slots))
               if sp not in matched_src]

    # reorder = relative order of matched devices changed (a plain shift of
    # absolute positions caused by an insertion/deletion is NOT a reorder)
    tgt_order = [primary[sp] for sp in sorted(primary)]
    reordered = list(tgt_order) != sorted(tgt_order)

    events: list[dict] = []
    for d in deleted:
        events.append({"type": "device_deleted", "position": d["position"],
                       "device": d["device"],
                       "detail": (f"device {d['device'] or 'unknown'} at source "
                                  f"position {d['position']} has no target counterpart")})
    for d in inserted:
        events.append({"type": "device_inserted", "position": d["position"],
                       "device": d["device"],
                       "detail": (f"device {d['device'] or 'unknown'} at target "
                                  f"position {d['position']} has no source counterpart")})
    if reordered:
        events.append({
            "type": "devices_reordered",
            "positions": [{"source": sp, "target": primary[sp]}
                          for sp in sorted(primary)],
            "detail": "relative order of matched devices changed",
        })
    for p in pairs:
        s, t = src.slots[p["source_position"]], tgt.slots[p["target_position"]]
        if s.ir_length != t.ir_length:
            events.append({
                "type": "ir_length_changed",
                "source_position": p["source_position"],
                "target_position": p["target_position"],
                "device": t.name or s.name,
                "from": s.ir_length, "to": t.ir_length,
                "detail": f"IR length {s.ir_length} -> {t.ir_length}",
            })
        if s.boundary != t.boundary:
            events.append({
                "type": "boundary_length_changed",
                "source_position": p["source_position"],
                "target_position": p["target_position"],
                "device": t.name or s.name,
                "from": s.boundary, "to": t.boundary,
                "detail": f"boundary length {s.boundary} -> {t.boundary}",
            })
        if (s.idcode or s.idcode_observed) != (t.idcode or t.idcode_observed):
            events.append({
                "type": "idcode_changed",
                "source_position": p["source_position"],
                "target_position": p["target_position"],
                "device": t.name or s.name,
                "from": _idcode_pair(src, tgt, p["source_position"],
                                     p["target_position"])["source"],
                "to": _idcode_pair(src, tgt, p["source_position"],
                                   p["target_position"])["target"],
                "detail": "IDCODE value/presence changed",
            })

    if ambiguous:
        status = "ambiguous"
    elif inserted or deleted:
        status = "partial"
    else:
        status = "unique"
    return {
        "status": status,
        "score": weight,
        "pairs": pairs,
        "inserted": inserted,
        "deleted": deleted,
        "reordered": reordered,
        "events": events,
        "enumeration_truncated": truncated,
        "candidate_alignments": [
            [{"source_position": sp, "target_position": a[sp]}
             for sp in sorted(a)]
            for a in alignments
        ] if ambiguous else [],
    }


# ---------------------------------------------------------------- register layouts

REGISTERS = ["IR", "DR:BYPASS", "DR:IDCODE", "DR:SAMPLE"]


@dataclass
class _RegSeg:
    position: int
    device: str | None
    length: int | None          # None = layout unknown from this segment on
    start: int | None           # first TDO-side chain bit; None once broken
    has_idcode: bool = False


def _layout(register: str, view: _VersionView) -> tuple[list[_RegSeg], int | None]:
    """TDO-out segment layout for one register.

    Returns (segments, total length). For SAMPLE, a device with unknown
    BOUNDARY_LENGTH breaks bit indexing: the segment and every segment closer
    to TDI get ``length``/``start`` of None and total length is None.
    """
    segs: list[_RegSeg] = []
    start = 0
    broken = False
    for tdo_i in range(len(view.chain) - 1, -1, -1):
        slot = view.chain[tdo_i]
        model = view.devices.get(slot.get("device") or "")
        has_id = False
        if register == "IR":
            length: int | None = slot["ir_length"]
        elif register == "DR:BYPASS":
            length = 1
        elif register == "DR:IDCODE":
            has_id = bool(model and model.get("idcode_value"))
            length = 32 if has_id else 1
        else:  # DR:SAMPLE
            length = model.get("boundary_length") if model else None
        if length is None:
            broken = True
        segs.append(_RegSeg(
            position=tdo_i,
            device=slot.get("device"),
            length=None if broken else length,
            start=None if broken else start,
            has_idcode=has_id,
        ))
        if not broken:
            start += length
    return segs, (None if broken else start)


def _seg_by_position(segs: list[_RegSeg]) -> dict[int, _RegSeg]:
    return {s.position: s for s in segs}


def _bit_row(register, src_seg, tgt_seg, sb, tb, local, status, reason,
             identity_change=False):
    return {
        "register": register,
        "status": status,
        "reason": reason,
        "source_bit": sb,
        "target_bit": tb,
        "source_position": src_seg.position if src_seg else None,
        "target_position": tgt_seg.position if tgt_seg else None,
        "device": (tgt_seg.device if tgt_seg and tgt_seg.device
                   else (src_seg.device if src_seg else None)),
        "register_bit": local,
        "identity_change": identity_change,
    }


def _length_reason(register: str) -> str:
    return {
        "IR": "ir_length_changed",
        "DR:SAMPLE": "boundary_length_changed",
    }.get(register, "register_length_changed")


def _register_mapping(register: str, src: _VersionView, tgt: _VersionView,
                      alignment: dict) -> dict:
    """Bit-index map of one register through the primary slot alignment."""
    src_segs, src_total = _layout(register, src)
    tgt_segs, tgt_total = _layout(register, tgt)
    sp_of = {p["source_position"]: p for p in alignment["pairs"]}
    src_by, tgt_by = _seg_by_position(src_segs), _seg_by_position(tgt_segs)

    rows: list[dict] = []
    seg_rows: list[dict] = []

    def seg_ref(seg: _RegSeg | None) -> dict | None:
        if seg is None or seg.start is None or seg.length is None:
            return None
        return {"position": seg.position, "device": seg.device,
                "start_bit": seg.start, "end_bit": seg.start + seg.length - 1,
                "length": seg.length}

    for ss in src_segs:
        pair = sp_of.get(ss.position)
        ts = tgt_by.get(pair["target_position"]) if pair else None
        identity_change = bool(pair and pair["match_type"] != "identical")
        sref, tref = seg_ref(ss), seg_ref(ts)
        if pair is None:
            seg_rows.append({"source": sref, "target": None,
                             "status": "dropped_segment",
                             "reasons": ["device_deleted"]})
            if ss.start is not None and ss.length:
                for lb in range(ss.length):
                    rows.append(_bit_row(
                        register, ss, None, ss.start + lb, None, lb,
                        "dropped", "device_deleted"))
            continue
        if sref is None or tref is None:
            seg_rows.append({"source": sref, "target": tref,
                             "status": "layout_unknown",
                             "reasons": ["boundary_length_unknown"]})
            continue
        overlap = min(ss.length, ts.length)
        reasons = []
        if ss.length != ts.length:
            reasons.append(_length_reason(register))
        if register == "DR:IDCODE" and ss.has_idcode != ts.has_idcode:
            reasons.append("idcode_register_added"
                           if ts.has_idcode else "idcode_register_removed")
        seg_status = "mapped" if not reasons else "partial"
        seg_rows.append({"source": sref, "target": tref,
                         "status": seg_status, "reasons": reasons})
        for lb in range(overlap):
            sb, tb = ss.start + lb, ts.start + lb
            if sb == tb and not identity_change and not reasons:
                status, reason = "unchanged", None
            elif identity_change and sb == tb:
                status, reason = "relabeled", pair["match_type"]
            else:
                status = "remapped"
                reason = reasons[0] if reasons else (
                    "chain_reordered" if alignment["reordered"]
                    else "position_shift")
            rows.append(_bit_row(register, ss, ts, sb, tb, lb,
                                 status, reason, identity_change))
        for lb in range(overlap, ss.length):
            reason = (_length_reason(register)
                      if register != "DR:IDCODE" or ss.has_idcode == ts.has_idcode
                      else "idcode_register_removed")
            rows.append(_bit_row(register, ss, ts, ss.start + lb, None, lb,
                                 "dropped", reason, identity_change))
        for lb in range(overlap, ts.length):
            reason = (_length_reason(register)
                      if register != "DR:IDCODE" or ss.has_idcode == ts.has_idcode
                      else "idcode_register_added")
            rows.append(_bit_row(register, ss, ts, None, ts.start + lb, lb,
                                 "added", reason, identity_change))

    # target segments with no source counterpart
    for ts in tgt_segs:
        if ts.position in {p["target_position"] for p in alignment["pairs"]}:
            continue
        tref = seg_ref(ts)
        seg_rows.append({"source": None, "target": tref,
                         "status": "added_segment",
                         "reasons": ["device_inserted"]})
        if ts.start is not None and ts.length:
            for lb in range(ts.length):
                rows.append(_bit_row(
                    register, None, ts, None, ts.start + lb, lb,
                    "added", "device_inserted"))

    changed = [r for r in rows if r["status"] != "unchanged"]
    max_len = max(src_total or 0, tgt_total or 0)
    full = max_len <= FULL_BITMAP_LIMIT and src_total is not None and tgt_total is not None
    return {
        "register": register,
        "mappable": src_total is not None and tgt_total is not None,
        "source_length": src_total,
        "target_length": tgt_total,
        "segments": seg_rows,
        "bits": rows if full else [],
        "full_bit_map": full,
        "changed_bits": changed,
        "counts": {
            "unchanged": sum(1 for r in rows if r["status"] == "unchanged"),
            "remapped": sum(1 for r in rows if r["status"] == "remapped"),
            "relabeled": sum(1 for r in rows if r["status"] == "relabeled"),
            "dropped": sum(1 for r in rows if r["status"] == "dropped"),
            "added": sum(1 for r in rows if r["status"] == "added"),
        },
    }


# ---------------------------------------------------------------- sample migration

def _register_of(kind: str, instruction: str | None) -> str | None:
    if kind == "ir":
        return "IR"
    inst = (instruction or "").strip().upper()
    if inst == "BYPASS":
        return "DR:BYPASS"
    if inst in ("IDCODE", "DEVICE_ID"):
        return "DR:IDCODE"
    if inst in ("SAMPLE", "PRELOAD"):
        return "DR:SAMPLE"
    return None


def _intervals(bits: list[int]) -> list[dict]:
    """Compress sorted bit indices into [start,end] ranges."""
    if not bits:
        return []
    out = []
    start = prev = bits[0]
    for b in bits[1:]:
        if b == prev + 1:
            prev = b
            continue
        out.append({"start_bit": start, "end_bit": prev})
        start = prev = b
    out.append({"start_bit": start, "end_bit": prev})
    return out


def _content_checks(view: _VersionView, register: str, tdo: str,
                    leading_missing: int, total: int) -> list[dict]:
    """Compare the captured content against what the target version expects.

    Returns a list of mismatch rows (empty = content consistent). Only bits
    actually present in ``tdo`` (accounting for the truncated capture head)
    are compared.
    """
    mismatches: list[dict] = []

    def observed(cb: int) -> str | None:
        idx = cb - leading_missing
        if idx < 0 or idx >= len(tdo):
            return None
        return tdo[idx]

    if register == "IR":
        segs, _ = _build_segments("ir", None, view.chain, view.devices)
        expected = sorted((cb, want, seg)
                          for seg in segs
                          for cb, want in seg.expected.items())
        for cb, want, seg in expected:
            val = observed(cb)
            if val is not None and val != want:
                mismatches.append({
                    "register": register, "target_bit": cb,
                    "target_position": seg.position, "device": seg.device,
                    "register_bit": cb - seg.start,
                    "tdo_index": cb - leading_missing,
                    "observed": val, "expected": want,
                    "reason": "ir_capture_mismatch",
                })
    elif register == "DR:BYPASS":
        for cb in range(total):
            val = observed(cb)
            if val == "1":
                mismatches.append({
                    "register": register, "target_bit": cb,
                    "target_position": None, "device": None,
                    "register_bit": cb, "tdo_index": cb - leading_missing,
                    "observed": "1", "expected": "0",
                    "reason": "bypass_bit_is_one",
                })
        segs, _ = _build_segments("dr", "BYPASS", view.chain, view.devices)
        for m in mismatches:
            seg = next((s for s in segs if s.start <= m["target_bit"]
                        < s.start + s.length), None)
            if seg:
                m["target_position"] = seg.position
                m["device"] = seg.device
                m["register_bit"] = m["target_bit"] - seg.start
    elif register == "DR:IDCODE":
        segs, _ = _build_segments("dr", "IDCODE", view.chain, view.devices)
        for seg in segs:
            model = view.devices.get(seg.device or "")
            if seg.length != 32 or not model:
                val = observed(seg.start)  # no-IDCODE device sits in BYPASS
                if val == "1":
                    mismatches.append({
                        "register": register, "target_bit": seg.start,
                        "target_position": seg.position, "device": seg.device,
                        "register_bit": 0,
                        "tdo_index": seg.start - leading_missing,
                        "observed": "1", "expected": "0",
                        "reason": "bypass_bit_is_one",
                    })
                continue
            chunk = ""
            complete = True
            for cb in range(seg.start, seg.start + 32):
                val = observed(cb)
                if val is None:
                    complete = False
                    break
                chunk += val
            if not complete:
                continue
            obs = (int(chunk, 2) if view.bit_order == "msb_first"
                   else int(chunk[::-1], 2))
            want = int(model["idcode_value"], 16)
            mask = int(model.get("idcode_mask") or "0xFFFFFFFF", 16)
            diff = (obs ^ want) & mask
            for k in range(32):
                if diff >> k & 1:
                    mismatches.append({
                        "register": register, "target_bit": seg.start + k,
                        "target_position": seg.position, "device": seg.device,
                        "register_bit": k,
                        "tdo_index": seg.start + k - leading_missing,
                        "observed": chunk[k],
                        "reason": "idcode_mismatch",
                    })
    return mismatches


def _echo_check(tdi: str, tdo: str, off: int) -> tuple[int, int]:
    """(matching, compared) echo bits at ``off``; -1 compared = a definite
    mismatch in the compared region; 0 = no bit comparable at this offset."""
    if off < 0 or off >= len(tdo):
        return 0, 0
    compared = matched = 0
    for i, b in enumerate(tdi):
        pos = off + i
        if pos >= len(tdo):
            break
        compared += 1
        if tdo[pos] != b:
            return -1, 0
        matched += 1
    return compared, matched


def _migrate_sample(sample: dict, src: _VersionView, tgt: _VersionView,
                    mappings: dict[str, dict]) -> dict:
    kind = sample["kind"]
    instruction = sample.get("instruction")
    register = _register_of(kind, instruction)
    tdi, tdo = sample["tdi"], sample["tdo"]

    _, src_total = _layout(register, src) if register else ([], None)
    _, tgt_total = _layout(register, tgt) if register else ([], None)

    item = {
        "origin": sample.get("origin", {}),
        "kind": kind,
        "instruction": instruction,
        "register": register or "wire",
        "tdo_length": len(tdo),
        "source_register_length": src_total,
        "target_register_length": tgt_total,
        "offset_observed": None,
        "leading_missing": 0,
        "verdict": None,
        "reasons": [],
        "interpretable": False,
        "remapped_bits": [],
        "dropped_bits": [],
        "added_bits": [],
        "missing_bits": {"count": 0, "intervals": []},
        "mismatch_bits": [],
        "affected_positions": [],
        "affected_devices": [],
    }
    fatal: list[str] = []
    candidates: list[tuple] = []

    if register is None:
        fatal.append("target_register_layout_unknown")
    elif tgt_total is None:
        fatal.append("boundary_length_unknown")
    elif tdo and len(set(tdo)) == 1:
        fatal.append("tdo_constant")
    else:
        # Try alignments at the target length and up to 4 bits early
        # (truncated capture head). Echo-contradicting offsets are ignored;
        # among the rest the one at the exact target length with the
        # strongest echo wins. If none fits but an echo alignment exists
        # near the target length, its content contradictions are reported
        # with full per-bit evidence (verdict contradicts_target).
        for k in (0, 1, 2, 3, 4):
            off = tgt_total - k
            if off < 0:
                continue
            compared, matched = _echo_check(tdi, tdo, off)
            if compared < 0:
                continue
            mism = _content_checks(tgt, register, tdo, k, tgt_total)
            if not mism and (k > 0 or compared > 0):
                candidates.append((k, compared, mism))
        if candidates:
            chosen = max(candidates, key=lambda c: (c[0] == 0, c[1], -c[0]))
            item["offset_observed"] = tgt_total - chosen[0]
            item["leading_missing"] = chosen[0]
        else:
            best, bscore = _best_offset(tdi, tdo)
            item["offset_observed"] = best
            if best is None or bscore < 4:
                fatal.append("alignment_failed")
            else:
                near = [k for k in range(5)
                        if tgt_total - k >= 0
                        and _echo_check(tdi, tdo, tgt_total - k)[0] >= 0]
                if near:
                    # best echo-aligned near-target offset: report its
                    # content contradictions below
                    k = min(near)
                    item["offset_observed"] = tgt_total - k
                    item["leading_missing"] = k
                    item["reasons"].append("content_contradicts_target")
                else:
                    fatal.append("register_length_mismatch")

    if fatal:
        item["verdict"] = "not_interpretable"
        item["reasons"] = sorted(set(item["reasons"] + fatal))
        return item

    # ---- content of the chosen alignment ----------------------------------
    k = item["leading_missing"]
    mapping = mappings[register]
    mismatches = _content_checks(tgt, register, tdo, k, tgt_total)
    item["mismatch_bits"] = mismatches

    # ---- bit migration through the structural map -------------------------
    observed_target: set[int] = set()
    for cb in range(tgt_total):
        idx = cb - k
        if 0 <= idx < len(tdo):
            observed_target.add(cb)
    missing = sorted(set(range(tgt_total)) - observed_target)
    item["missing_bits"] = {"count": len(missing), "intervals": _intervals(missing)}

    for r in mapping["changed_bits"]:
        entry = {k: r[k] for k in (
            "register", "source_bit", "target_bit", "source_position",
            "target_position", "device", "register_bit", "reason")}
        if r["status"] in ("remapped", "relabeled"):
            item["remapped_bits"].append(entry)
        elif r["status"] == "dropped":
            item["dropped_bits"].append(entry)
        elif r["status"] == "added":
            item["added_bits"].append(entry)

    # ---- affected registers / devices ------------------------------------
    affected = (item["remapped_bits"] + item["dropped_bits"]
                + item["added_bits"] + mismatches)
    item["affected_positions"] = sorted({
        p for e in affected
        for p in (e.get("target_position"), e.get("source_position"))
        if p is not None})
    item["affected_devices"] = sorted({
        e["device"] for e in affected if e.get("device")})

    if mismatches:
        item["verdict"] = "contradicts_target"
        item["reasons"] = sorted(set(
            item["reasons"] + [m["reason"] for m in mismatches]))
        item["interpretable"] = False
    elif missing:
        item["verdict"] = "interpretable_with_gaps"
        item["reasons"] = sorted(set(
            item["reasons"] + ["capture_head_or_tail_truncated"]))
        item["interpretable"] = True
    else:
        item["verdict"] = "interpretable"
        item["interpretable"] = True
    return item


# ---------------------------------------------------------------- entry point

def compare_versions(source: dict, target: dict, samples: list[dict],
                     source_candidate: int = 0,
                     target_candidate: int = 0) -> dict:
    """Build the full diff report.

    ``source``/``target`` are rows from ``db.get_version``; ``samples`` are
    dicts with kind/instruction/tdi/tdo plus an optional ``origin`` marker.
    Raises ValueError on an out-of-range candidate index.
    """
    src = _view(source, source_candidate)
    tgt = _view(target, target_candidate)
    if src.session != tgt.session:
        raise ValueError(
            f"versions belong to different sessions: {src.session!r} vs "
            f"{tgt.session!r}")

    alignment = _align(src, tgt)
    mappings = {r: _register_mapping(r, src, tgt, alignment) for r in REGISTERS}

    items = [_migrate_sample(s, src, tgt, mappings) for s in samples]

    # per-instruction rollup
    instr_summary: list[dict] = []
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        groups.setdefault((it["kind"], it["instruction"]), []).append(it)
    for (kind, inst), grp in sorted(groups.items()):
        verdicts: dict[str, int] = {}
        for it in grp:
            verdicts[it["verdict"]] = verdicts.get(it["verdict"], 0) + 1
        instr_summary.append({
            "kind": kind,
            "instruction": inst,
            "register": grp[0]["register"],
            "samples": len(grp),
            "verdicts": verdicts,
            "all_interpretable": all(i["interpretable"] for i in grp),
            "reasons": sorted({r for i in grp for r in i["reasons"]}),
            "affected_devices": sorted({
                d for i in grp for d in i["affected_devices"]}),
            "affected_positions": sorted({
                p for i in grp for p in i["affected_positions"]}),
        })

    # per-consistency-batch rollup (only for samples coming from batches)
    batch_summary: list[dict] = []
    bgroups: dict[int, list[dict]] = {}
    for it in items:
        bid = it["origin"].get("consistency_batch_id")
        if bid is not None:
            bgroups.setdefault(bid, []).append(it)
    for bid, grp in sorted(bgroups.items()):
        batch_summary.append({
            "consistency_batch_id": bid,
            "runs": len(grp),
            "labels": [i["origin"].get("label") for i in grp],
            "interpretable": sum(1 for i in grp if i["interpretable"]),
            "not_interpretable": sum(
                1 for i in grp if i["verdict"] == "not_interpretable"),
            "contradictions": sum(
                1 for i in grp if i["verdict"] == "contradicts_target"),
        })

    n = lambda what: sum(1 for e in alignment["events"] if e["type"] == what)
    return {
        "session": src.session,
        "source": {
            "version_id": src.version_id, "candidate": source_candidate,
            "bit_order": src.bit_order,
            "slots": len(src.slots),
            "ir_length": sum(s.ir_length for s in src.slots),
        },
        "target": {
            "version_id": tgt.version_id, "candidate": target_candidate,
            "bit_order": tgt.bit_order,
            "slots": len(tgt.slots),
            "ir_length": sum(s.ir_length for s in tgt.slots),
        },
        "summary": {
            "alignment_status": alignment["status"],
            "inserted": len(alignment["inserted"]),
            "deleted": len(alignment["deleted"]),
            "reordered": alignment["reordered"],
            "ir_length_changes": n("ir_length_changed"),
            "boundary_length_changes": n("boundary_length_changed"),
            "idcode_changes": n("idcode_changed"),
            "samples_total": len(items),
            "samples_interpretable": sum(1 for i in items if i["interpretable"]),
            "samples_contradicting": sum(
                1 for i in items if i["verdict"] == "contradicts_target"),
            "samples_not_interpretable": sum(
                1 for i in items if i["verdict"] == "not_interpretable"),
            "instructions_not_explainable": [
                f"{g['kind']}:{g['instruction'] or '-'}"
                for g in instr_summary if not g["all_interpretable"]],
        },
        "alignment": alignment,
        "register_mappings": [mappings[r] for r in REGISTERS],
        "sample_migration": {
            "selection": {},  # filled by the API layer
            "items": items,
            "instruction_summary": instr_summary,
            "batch_summary": batch_summary,
        },
    }
