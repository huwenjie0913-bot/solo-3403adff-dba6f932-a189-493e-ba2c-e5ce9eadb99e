"""Repeat-sample consistency analysis and intermittent-fault localization.

Multiple labeled IR/DR samples of the *same* session are grouped by
(kind, instruction). Each run is aligned on its TDI echo; the TDO capture
bits are then mapped through a previously saved chain version. For every
chain bit the module records the stable value, the number of toggles across
runs and the runs/intervals where the bit was not observed.

Runs that cannot be aligned (TDO constant, register-length mismatch, echo
not found) are reported with the exact run and TDO positions and are
excluded from the per-bit statistics. Raw samples and inferred versions are
never modified.

Chain convention (same as ``inference.py``): position 0 is closest to TDI;
TDO bit 0 is the first bit clocked out and belongs to the device closest to
TDO. Register-local bit 0 is the bit closest to TDO within that register.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .inference import find_offsets

# Echo starting up to this many bits early means the capture head was
# truncated (leading capture bits missing) rather than a length mismatch.
LEADING_MISSING_TOLERANCE = 4
MIN_ECHO_MATCH = 4       # minimum matching bits that prove an echo alignment
MIN_SHIFT_FIT = 4        # minimum comparable bits for an in-region shift
SHIFT_FIT_RATIO = 0.9    # fraction of explained bits for a "fixed offset" call
SHIFT_SEARCH = (-2, -1, 1, 2)


# ---------------------------------------------------------------- chain mapping

@dataclass
class _Seg:
    index: int
    position: int | None            # chain position, TDI-side (None = wire mode)
    device: str | None
    register: str
    length: int
    start: int                      # first TDO-side chain bit of the segment
    expected: dict[int, str] = field(default_factory=dict)  # chain bit -> 0/1


def _build_segments(kind: str, instruction: str | None, chain: list[dict],
                    devices: dict[str, dict]):
    """Return (segments TDO-side order, total register length).

    ``(None, None)`` means the DR layout cannot be derived from the saved
    version (unknown boundary length / unknown instruction): the group is
    analyzed in wire mode, without device mapping.
    """
    norm = (instruction or "").upper()
    n = len(chain)
    lengths: list[int] = []
    register = ""

    if kind == "ir":
        register = "IR"
        lengths = [s["ir_length"] for s in chain]
    elif norm == "BYPASS":
        register = "DR:BYPASS"
        lengths = [1] * n
    elif norm in ("IDCODE", "DEVICE_ID"):
        register = "DR:IDCODE"
        for s in chain:
            model = devices.get(s.get("device") or "")
            lengths.append(32 if model and model.get("idcode_value") else 1)
    elif norm in ("SAMPLE", "PRELOAD"):
        register = f"DR:{norm}"
        for s in chain:
            model = devices.get(s.get("device") or "")
            bl = model.get("boundary_length") if model else None
            if bl is None:
                return None, None
            lengths.append(bl)
    else:
        return None, None

    segs: list[_Seg] = []
    start = 0
    # TDO bit 0.. maps onto the saved chain in TDO-consumption order, which
    # is the same order inference.py uses (candidate chain is TDI-side first,
    # but capture segments are consumed TDO-side first): the first segment is
    # the device at TDO-side = the last chain position.
    for tdo_i in range(n - 1, -1, -1):
        slot = chain[tdo_i]
        length = lengths[tdo_i]
        seg = _Seg(
            index=len(segs),
            position=tdo_i,
            device=slot.get("device"),
            register=register,
            length=length,
            start=start,
        )
        if kind == "ir":
            model = devices.get(slot.get("device") or "")
            if model and model.get("ir_capture"):
                cap = model["ir_capture"][::-1]  # BSDL order -> shift-out order
                for k, ch in enumerate(cap[:length]):
                    if ch in "01":
                        seg.expected[start + k] = ch
        segs.append(seg)
        start += length
    return segs, sum(lengths)


def _segment_of(segs: list[_Seg] | None, cb: int) -> _Seg | None:
    if not segs:
        return None
    for seg in segs:  # segments are small and ordered
        if seg.start <= cb < seg.start + seg.length:
            return seg
    return None


# ---------------------------------------------------------------- alignment

def _echo_score(tdi: str, tdo: str, off: int) -> tuple[int, int]:
    """(matching, compared) bits of the TDI prefix against TDO at ``off``."""
    compared = matched = 0
    for i, b in enumerate(tdi):
        if off + i >= len(tdo):
            break
        compared += 1
        if tdo[off + i] == b:
            matched += 1
    return matched, compared


def _best_offset(tdi: str, tdo: str) -> tuple[int | None, int]:
    """Start position of the TDI echo.

    ``inference.find_offsets`` finds every offset where the 32-bit TDI probe
    matches, including zero-length "matches" at/after the TDO tail (only a
    handful of bits compared). Rank candidates with the *whole* TDI stream by
    matching bits, earliest offset on ties; fall back to a full-stream scan
    when the probe never matches.
    """
    candidates = find_offsets(tdi, tdo, set())
    if candidates:
        scored = sorted(
            ((*_echo_score(tdi, tdo, off), -off) for off in candidates),
            reverse=True)
        matched, _, neg_off = scored[0]
        return -neg_off, matched
    best: tuple[int, float, int] | None = None  # (matched, ratio, -offset)
    for off in range(len(tdo)):
        matched, compared = _echo_score(tdi, tdo, off)
        if matched < MIN_ECHO_MATCH:
            continue
        ratio = matched / compared
        key = (matched, ratio, -off)
        if best is None or key > best:
            best = key
    if best is None:
        return None, 0
    return -best[2], best[0]


# ---------------------------------------------------------------- per-group analysis

def _analyze_group(key: tuple[str, str], grp: list[dict], chain: list[dict],
                   devices: dict[str, dict]) -> dict:
    kind, inst_key = key
    instruction = None if inst_key == "" else inst_key
    segs, total_length = _build_segments(kind, instruction, chain, devices)
    mapped = segs is not None

    # ---- per-run alignment -------------------------------------------------
    run_rows: list[dict] = []
    aligned: list[dict] = []
    diagnoses: list[dict] = []
    constant_runs: list[dict] = []

    for r in grp:
        tdi, tdo = r["tdi"], r["tdo"]
        row = {
            "label": r["label"],
            "run_index": r["run_index"],
            "status": "aligned",
            "aligned": False,
            "excluded_reason": None,
            "register_length_expected": total_length,
            "offset_observed": None,
            "leading_missing": 0,
            "capture_bits": None,
            "tdo_length": len(tdo),
        }
        if tdo and len(set(tdo)) == 1:
            row.update(status="excluded", excluded_reason="tdo_constant",
                       offset_observed=None)
            constant_runs.append((r, row))
        else:
            best, score = _best_offset(tdi, tdo)
            row["offset_observed"] = best
            if best is None or score < MIN_ECHO_MATCH:
                row.update(status="excluded", excluded_reason="alignment_failed")
                diagnoses.append(_mismatch_diag(
                    r, row, mapped, total_length, best, segs,
                    kind="alignment_failed",
                    detail="TDI echo not found in TDO (alignment failed; "
                           "check wiring/probing or exclude this run)"))
            elif mapped:
                if best == total_length:
                    row["capture_bits"] = total_length  # perfectly aligned
                elif total_length - LEADING_MISSING_TOLERANCE <= best < total_length:
                    row["leading_missing"] = total_length - best
                    row["capture_bits"] = best
                else:
                    row.update(status="excluded",
                               excluded_reason="register_length_mismatch")
                    diagnoses.append(_mismatch_diag(
                        r, row, mapped, total_length, best, segs,
                        kind="register_length_mismatch",
                        detail=(f"register length {total_length} expected from "
                                f"saved chain, TDI echo starts at TDO bit "
                                f"{best}"),
                        match_end=min(best + score, len(tdo))))
            else:
                # wire mode: the capture region is everything before the echo
                row["capture_bits"] = best
            # wire mode: any echo alignment is accepted
        if row["status"] == "aligned":
            row["aligned"] = True
            aligned.append((r, row))
        run_rows.append(row)

    if constant_runs:
        diagnoses.append({
            "kind": "tdo_constant",
            "runs": [r["label"] for r, _ in constant_runs],
            "affected_positions": [s.position for s in segs] if mapped else [],
            "affected_devices": [s.device for s in segs] if mapped else [],
            "register_bits": [],
            "first_anomaly": _run_ref(grp, constant_runs[0][0]["label"], 0),
            "detail": "TDO is constant for the whole scan: open chain, "
                      "shorted TDO or TAP held in reset",
            "evidence": [
                {"run": rr["label"],
                 "tdo_indices": [0, rr["tdo_length"] - 1],
                 "chain_bits": []}
                for _, rr in constant_runs
            ],
            "_sort": (-1, constant_runs[0][0]["run_index"], 0),
        })

    # ---- bit universe ------------------------------------------------------
    if mapped:
        bit_count = total_length
    elif aligned:
        bit_count = max(rr["capture_bits"] - rr["leading_missing"]
                        for _, rr in aligned)
        bit_count = max(bit_count, 0)
    else:
        bit_count = 0

    # values[cb] = list in run order of (run, row, TDO index, value).
    # Mapped mode chains capture bits onto logical bits from the TDO side
    # (k leading bits were truncated); wire mode uses the raw capture region.
    values: list[list[tuple]] = [[] for _ in range(bit_count)]
    for r, rr in aligned:
        k = rr["leading_missing"]
        if mapped:
            for cb in range(bit_count):
                idx = cb - k
                if 0 <= idx < len(r["tdo"]):
                    values[cb].append((r, rr, idx, r["tdo"][idx]))
        else:
            n = rr["capture_bits"]
            for idx in range(n):
                values[idx].append((r, rr, idx, r["tdo"][idx]))

    aligned_labels = [r["label"] for r, _ in aligned]
    bits: list[dict] = []
    unstable: list[int] = []
    missing_intervals: list[dict] = []

    for cb in range(bit_count):
        obs = values[cb]
        seg = _segment_of(segs, cb) if mapped else None
        entry = {
            "chain_bit": cb,
            "segment": seg.index if seg else None,
            "position": seg.position if seg else None,
            "device": seg.device if seg else None,
            "register": seg.register if seg else "wire",
            "register_bit": (cb - seg.start) if seg else cb,
            "observed_runs": [o[0]["label"] for o in obs],
            "missing_runs": [lab for lab in aligned_labels
                             if lab not in {o[0]["label"] for o in obs}],
            "count_0": sum(1 for o in obs if o[3] == "0"),
            "count_1": sum(1 for o in obs if o[3] == "1"),
            "toggles": 0,
            "stable": None,
            "anomalous": False,
            "expected": (seg.expected.get(cb) if seg and cb in seg.expected
                         else None),
            "values": [
                {"run": o[0]["label"], "run_index": o[0]["run_index"],
                 "tdo_index": o[2], "value": o[3]}
                for o in obs
            ],
        }
        if obs:
            entry["toggles"] = sum(
                1 for a, b in zip(obs, obs[1:]) if a[3] != b[3])
            c0, c1 = entry["count_0"], entry["count_1"]
            if c0 == c1:
                entry["stable"] = None  # evenly split, no stable value
            else:
                entry["stable"] = "0" if c0 > c1 else "1"
        if len(obs) >= 2 and 0 < entry["count_0"] < len(obs):
            entry["anomalous"] = True
            unstable.append(cb)
        bits.append(entry)

    missing_intervals = _missing_intervals(bits)

    # ---- in-region fixed-offset (coherent shift) explanation --------------
    # Only meaningful in mapped mode: wire mode is anchored on the echo, so
    # a shifted capture simply presents as a different echo offset.
    explained: dict[int, set[str]] = {}   # chain bit -> runs explained by shift
    if mapped:
        for r, rr in aligned:
            anom_bits = [cb for cb in unstable if bits[cb]["stable"] is not None
                         and r["label"] in _minority_runs(bits[cb])]
            if not anom_bits:
                continue
            ref = _reference_values(bits, r["label"])
            k = rr["leading_missing"]
            best_fit = None
            for d in SHIFT_SEARCH:
                total = matched = 0
                matched_bits = []
                for cb, refv in ref.items():
                    idx = cb - k - d
                    if 0 <= idx < len(r["tdo"]):
                        total += 1
                        if r["tdo"][idx] == refv:
                            matched += 1
                            matched_bits.append(cb)
                if total >= MIN_SHIFT_FIT and matched / total >= SHIFT_FIT_RATIO:
                    if best_fit is None or matched > best_fit[1]:
                        best_fit = (d, matched, total, matched_bits)
            if best_fit:
                d, matched, total, matched_bits = best_fit
                # anomalous bits covered by the shift are explained and
                # removed from the residual classification pool
                explained_bits = [cb for cb in matched_bits if cb in anom_bits]
                for cb in explained_bits:
                    explained.setdefault(cb, set()).add(r["label"])
                diagnoses.append(_shift_diag(
                    r, rr, d, len(explained_bits), total, explained_bits,
                    segs, bits))

    pool = [cb for cb in unstable
            if explained.get(cb, set()) != _minority_runs(bits[cb])]

    # ---- classify the remaining anomalous bits ----------------------------
    diagnoses.extend(_classify(pool, bits, segs, mapped))
    diagnoses.sort(key=lambda d: d["_sort"])
    for d in diagnoses:
        d.pop("_sort", None)

    seg_rows = [
        {
            "segment": s.index,
            "position": s.position,
            "device": s.device,
            "register": s.register,
            "length": s.length,
            "start_bit": s.start,
            "end_bit": s.start + s.length - 1,
            "bit_numbering": "local bit 0 = closest to TDO (first bit out)",
            "expected_bits": [
                {"chain_bit": cb, "register_bit": cb - s.start, "value": v}
                for cb, v in sorted(s.expected.items())
            ],
        }
        for s in (segs or [])
    ]

    return {
        "kind": kind,
        "instruction": instruction,
        "register": segs[0].register if mapped else "wire",
        "mapping": "mapped" if mapped else "wire",
        "register_length": total_length if mapped else None,
        "runs": run_rows,
        "run_labels": [r["label"] for r in grp],
        "segments": seg_rows,
        "missing_intervals": missing_intervals,
        "bits": bits,
        "diagnoses": diagnoses,
        "summary": {
            "runs": len(grp),
            "aligned": len(aligned),
            "excluded": len(grp) - len(aligned),
            "bits_total": bit_count,
            "stable_bits": sum(1 for b in bits if not b["anomalous"]),
            "unstable_bits": len(unstable),
            "missing_bit_occurrences": sum(
                len(b["missing_runs"]) for b in bits),
        },
    }


# ---------------------------------------------------------------- diagnosis helpers

def _minority_runs(bit: dict) -> set[str]:
    stable = bit["stable"]
    if stable is None:
        # tie: call the first observed value's opposition the minority set
        stable = bit["values"][0]["value"] if bit["values"] else None
    return {v["run"] for v in bit["values"] if v["value"] != stable}


def _reference_values(bits: list[dict], skip_run: str) -> dict[int, str]:
    """Majority value at each bit, taken from runs other than ``skip_run``."""
    ref: dict[int, str] = {}
    for b in bits:
        vals = [v["value"] for v in b["values"] if v["run"] != skip_run]
        if not vals:
            continue
        c0 = vals.count("0")
        c1 = len(vals) - c0
        if c0 != c1:
            ref[b["chain_bit"]] = "0" if c0 > c1 else "1"
    return ref


def _run_ref(grp: list[dict], label: str, tdo_index: int) -> dict:
    r = next(x for x in grp if x["label"] == label)
    return {"run": label, "run_index": r["run_index"], "tdo_index": tdo_index}


def _witness(bits: list[dict], cb_list: list[int], segs, run: str | None = None) -> dict:
    """Raw index evidence supporting a conclusion."""
    per_run: dict[str, dict] = {}
    for cb in cb_list:
        for v in bits[cb]["values"]:
            if run is not None and v["run"] != run:
                continue
            e = per_run.setdefault(v["run"], {"run": v["run"], "tdo_indices": [],
                                              "chain_bits": []})
            e["tdo_indices"].append(v["tdo_index"])
            e["chain_bits"].append(cb)
    for e in per_run.values():
        e["tdo_indices"] = sorted(set(e["tdo_indices"]))
        e["chain_bits"] = sorted(set(e["chain_bits"]))
    return {"bits": [
        {"chain_bit": cb,
         "position": (s.position if (s := _segment_of(segs, cb)) else None),
         "device": s.device if s else None,
         "register": s.register if s else "wire",
         "register_bit": (cb - s.start) if s else cb}
        for cb in cb_list
    ], "samples": list(per_run.values())}


def _mismatch_diag(r: dict, row: dict, mapped: bool, total, best, segs,
                   detail: str, match_end: int | None = None,
                   kind: str = "register_length_mismatch") -> dict:
    evidence = []
    if best is not None:
        end = match_end if match_end is not None else min(best + 1, row["tdo_length"])
        evidence.append({"run": r["label"],
                         "tdo_indices": [best, max(best, end - 1)],
                         "chain_bits": []})
    return {
        "kind": kind,
        "runs": [r["label"]],
        "register_length_expected": total if mapped else None,
        "offset_observed": best,
        "affected_positions": [s.position for s in segs] if mapped else [],
        "affected_devices": [s.device for s in segs] if mapped else [],
        "register_bits": [],
        "first_anomaly": {"run": r["label"], "run_index": r["run_index"],
                          "tdo_index": best if best is not None else 0},
        "detail": detail,
        "evidence": {"bits": [], "samples": evidence},
        "_sort": (0, r["run_index"], 0),
    }


def _shift_diag(r, rr, d, matched, total, matched_bits, segs, bits) -> dict:
    affected = sorted({s.index for cb in matched_bits
                       if (s := _segment_of(segs, cb))})
    seg_objs = [segs[i] for i in affected] if segs else []
    first_anom = None
    for cb in matched_bits:
        for v in bits[cb]["values"]:
            if v["run"] == r["label"]:
                first_anom = {"run": r["label"], "run_index": r["run_index"],
                              "chain_bit": cb, "tdo_index": v["tdo_index"],
                              "value": v["value"],
                              "stable": bits[cb]["stable"]}
                break
        if first_anom:
            break
    direction = "later" if d > 0 else "earlier"
    return {
        "kind": "segment_fixed_offset",
        "runs": [r["label"]],
        "shift_bits": d,
        "matched_bits": matched,
        "compared_bits": total,
        "affected_positions": [s.position for s in seg_objs],
        "affected_devices": [s.device for s in seg_objs],
        "affected_segments": affected,
        "register_bits": sorted(matched_bits),
        "first_anomaly": first_anom,
        "detail": (f"capture content of run {r['label']!r} matches the other "
                   f"runs shifted by {abs(d)} bit(s) ({direction}): a fixed "
                   f"offset ({matched}/{total} bits explained); run excluded "
                   f"from those bits"),
        "evidence": _witness(bits, sorted(matched_bits), segs, r["label"]),
        "_sort": (1, r["run_index"], (matched_bits[0] if matched_bits else 0)),
    }


def _classify(pool: list[int], bits: list[dict], segs, mapped: bool) -> list[dict]:
    if not pool:
        return []

    # Global single-run anomaly: every unstable bit is carried by the same
    # sole run across more than one segment -> one bad isolated capture.
    minority = {cb: _minority_runs(bits[cb]) for cb in pool}
    unique_runs = {lab for s in minority.values() for lab in s}
    if (len(unique_runs) == 1 and len(minority) >= 2
            and len({(_segment_of(segs, cb).index if mapped else None)
                     for cb in pool}) > 1):
        only = next(iter(unique_runs))
        return [_pool_diag("isolated_sample_anomaly", [only], pool, bits, segs,
                           detail=(f"only run {only!r} deviates and the bad "
                                   "bits span multiple devices: isolated "
                                   "sampling anomaly of that run"))]

    out: list[dict] = []
    seg_ids = sorted({(_segment_of(segs, cb).index if mapped else None)
                      for cb in pool})
    for sid in seg_ids:
        seg_bits = [cb for cb in pool
                    if ((_segment_of(segs, cb).index if mapped else None) == sid)]
        # several runs carry the flip -> intermittent; the rest are
        # single-minority bits assigned per run below
        multi = [cb for cb in seg_bits if len(minority[cb]) >= 2]
        per_run: dict[str, list[int]] = {}
        for cb in seg_bits:
            mins = minority[cb]
            if len(mins) == 1 and cb not in multi:
                per_run.setdefault(next(iter(mins)), []).append(cb)
        # intermittent device: a bit flips in several independent runs;
        # a single run (even with several bad bits in this device) is an
        # isolated sampling anomaly of that run, not device intermittency
        inter = sorted(multi)
        isolated = sorted(per_run.items())
        if inter:
            # runs that actually read the anomalous value on at least one bit
            bad_runs = sorted({lab for cb in inter for lab in minority[cb]})
            seg = segs[sid] if mapped and sid is not None else None
            detail = (
                f"intermittent toggling on {seg.device if seg else 'wire'}"
                f" ({seg.register if seg else 'wire'} bits "
                f"{sorted(cb - seg.start for cb in inter) if seg else sorted(inter)}); "
                "other runs read the same bit stably")
            out.append(_pool_diag("intermittent_device_toggle", bad_runs,
                                  sorted(inter), bits, segs, detail))
        for lab, cbs in isolated:
            detail = (f"single bit deviates in run {lab!r} only: isolated "
                      "sampling anomaly (probe glitch / single read error)")
            out.append(_pool_diag("isolated_sample_anomaly", [lab],
                                  sorted(cbs), bits, segs, detail))
    return out


def _pool_diag(kind: str, runs: list[str], cb_list: list[int],
               bits: list[dict], segs, detail: str) -> dict:
    affected = sorted({s.index for cb in cb_list
                       if (s := _segment_of(segs, cb))})
    seg_objs = [segs[i] for i in affected] if segs else []
    first = None
    candidates = []
    for cb in cb_list:
        mins = _minority_runs(bits[cb])
        for v in bits[cb]["values"]:
            if v["run"] in runs and v["run"] in mins:
                candidates.append((v["run_index"], cb, v["tdo_index"],
                                   v["value"], v["run"]))
    if candidates:
        ri, cb, tdo_idx, value, run = min(candidates)
        first = {"run": run, "run_index": ri, "chain_bit": cb,
                 "tdo_index": tdo_idx, "value": value,
                 "stable": bits[cb]["stable"]}
    base = 2 if kind == "intermittent_device_toggle" else 3
    return {
        "kind": kind,
        "runs": runs,
        "affected_positions": [s.position for s in seg_objs],
        "affected_devices": [s.device for s in seg_objs],
        "affected_segments": affected,
        "register_bits": sorted(cb_list),
        "first_anomaly": first,
        "detail": detail,
        "evidence": _witness(bits, cb_list, segs),
        "_sort": (base,
                  first["run_index"] if first else 10**9,
                  cb_list[0] if cb_list else 0),
    }


def _missing_intervals(bits: list[dict]) -> list[dict]:
    """Contiguous chain-bit ranges missing in the same aligned run set."""
    intervals = []
    start = None
    prev_sig = None
    for i, b in enumerate(bits):
        sig = tuple(b["missing_runs"])
        if sig:
            if start is None or sig != prev_sig:
                if start is not None:
                    intervals[-1]["end_bit"] = i - 1
                start = i
                intervals.append({"start_bit": i, "end_bit": i,
                                  "runs": list(sig)})
            else:
                intervals[-1]["end_bit"] = i
        else:
            start = None
        prev_sig = sig
    return intervals


# ---------------------------------------------------------------- entry point

def analyze_consistency(runs: list[dict], version: dict,
                        candidate_index: int = 0) -> dict:
    """Analyze labeled repeat samples against a saved chain version.

    ``runs`` are dicts with label/kind/instruction/tdi/tdo; ``version`` is a
    row as returned by ``db.get_version``. Raises ValueError when the
    candidate index is out of range.
    """
    result = version["result"]
    cands = result.get("candidates", [])
    if candidate_index >= len(cands):
        raise ValueError(
            f"candidate {candidate_index} not found ({len(cands)} available)")
    chain = cands[candidate_index]["chain"]
    devices = {d["name"]: d for d in result.get("devices", [])}
    bit_order = version["request"].get("bit_order", "lsb_first")

    ordered = [dict(r, run_index=i) for i, r in enumerate(runs)]
    groups: dict[tuple[str, str], list[dict]] = {}
    group_order: list[tuple[str, str]] = []
    for r in ordered:
        key = (r["kind"], (r["instruction"] or "").strip().upper())
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(r)

    return {
        "version_id": version["id"],
        "candidate": candidate_index,
        "session": version["session"],
        "bit_order": bit_order,
        "run_count": len(ordered),
        "groups": [_analyze_group(k, groups[k], chain, devices)
                   for k in group_order],
    }
