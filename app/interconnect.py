"""Board-level boundary-scan interconnect test planning and analysis.

Two operations are offered, both strictly offline (no hardware access):

* :func:`plan_interconnect` — from a *saved* chain version (candidate chain +
  device BSDL models, including the parsed BOUNDARY_REGISTER) and a board
  netlist request, select exactly one active driver per net, put every other
  driver of the net into Hi-Z via its BSDL control cell/disable value, and
  emit conflict-free EXTEST boundary-register vectors (all-0 / walking-1 /
  all-1). Nets that cannot be driven safely — unknown devices, no usable
  BSDL model, no observer, extra drivers that cannot be tri-stated, or
  control cells shared with other nets/unmanaged pins — are skipped with an
  explicit reason and never contribute executable vectors.

* :func:`analyze_interconnect` — take the measured TDO of every vector of a
  saved plan, unload the chain bits onto the observer pins, and diagnose
  each net as pass / suspected open / fixed level, plus bridging candidates
  whose captured word equals, inverts, wire-combines or merely toggles in
  sync with another net's driven word. Every finding is located to the net,
  device pin (chain slot + port + boundary cell + chain bit) and vector.

Chain convention (identical to ``inference.py``/``consistency.py``): chain
position 0 is closest to TDI; TDO bit 0 is the first bit clocked out. Inside
one device, BSDL boundary cell 0 is the cell closest to TDO, so a cell's
shift-order bit index equals its BSDL cell number; its global chain bit is
the device's TDO-side offset plus the cell number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------- skip reasons

SKIP_NO_DRIVER = "no_driver"
SKIP_NO_OBSERVER = "no_observer"
SKIP_UNKNOWN_DEVICE = "unknown_device"
SKIP_MODEL_UNAVAILABLE = "boundary_model_unavailable"
SKIP_AMBIGUOUS_DEVICE = "ambiguous_device_in_chain"
SKIP_ENDPOINT_NOT_FOUND = "endpoint_pin_not_found"
SKIP_CANNOT_TRISTATE = "cannot_safely_tristate"
SKIP_CONTROL_CONFLICT = "control_conflict"
SKIP_CHAIN_NOT_EXECUTABLE = "chain_not_executable"


# ----------------------------------------------------------------- helpers

def _canon_port(port: str) -> str:
    """Canonicalize a BSDL port identifier: upper-case, no whitespace,
    bracket indexing normalized to parentheses (``DATA[3]`` -> ``DATA(3)``)."""
    p = "".join(port.split()).upper().replace("[", "(").replace("]", ")")
    return p


@dataclass(eq=False)
class _SlotView:
    position: int
    name: str | None              # None = unidentified chain slot
    boundary_length: int | None
    cells: list[dict] | None      # parsed boundary cells, None = unavailable
    model_source: str | None = None  # "version" | "device_library"
    offset: int | None = None     # TDO-side chain-bit offset
    # ccell number -> list of output3/bidir rows controlling it
    controls: dict[int, list[dict]] = field(default_factory=dict)


@dataclass(eq=False)
class _Pin:
    slot: _SlotView
    port: str                     # canonical port id
    driver_rows: list[dict] = field(default_factory=list)
    observer_rows: list[dict] = field(default_factory=list)


@dataclass(eq=False)
class _Net:
    name: str
    raw: dict
    pins: list[_Pin] = field(default_factory=list)
    status: str = "planned"
    skip_reason: str | None = None
    skip_detail: str = ""
    driver: dict | None = None
    disabled: list[dict] = field(default_factory=list)
    forced_active: list[dict] = field(default_factory=list)
    observers: list[dict] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)


def _is_driver(row: dict) -> bool:
    return "output" in row["capabilities"]


def _is_observer(row: dict) -> bool:
    return "input" in row["capabilities"]


def _is_tristate(row: dict) -> bool:
    return "tristate" in row["capabilities"]


def _pin_ref(slot: _SlotView, port: str, cell: int | None = None) -> dict:
    return {"device": slot.name, "position": slot.position, "port": port,
            "cell": cell}


def _chain_bit(slot: _SlotView, cell: int, executable: bool) -> int | None:
    if not executable or slot.offset is None:
        return None
    return slot.offset + cell


# ----------------------------------------------------------------- plan build

def _resolve_slots(chain: list[dict], devices: dict[str, dict],
                   library: dict[str, list[dict]]):
    """Build slot views with boundary cells. Returns (slots, warnings)."""
    name_counts: dict[str, int] = {}
    for s in chain:
        if s.get("device"):
            name_counts[s["device"]] = name_counts.get(s["device"], 0) + 1
    ambiguous = {n for n, c in name_counts.items() if c > 1}

    slots: list[_SlotView] = []
    warnings: list[str] = []
    for s in chain:
        name = s.get("device")
        model = devices.get(name or "")
        bl = model.get("boundary_length") if model else None
        cells = model.get("boundary_cells") if model else None
        source = "version" if cells else None
        if name and not cells:
            matches = [m for m in library.get(name, [])
                       if m.get("boundary_cells")
                       and (bl is None or m.get("boundary_length") == bl)]
            if matches:
                chosen = matches[-1]
                cells = chosen["boundary_cells"]
                bl = chosen.get("boundary_length", bl)
                source = "device_library"
                warnings.append(
                    f"device {name!r} (chain position {s['position']}): "
                    f"BOUNDARY_REGISTER taken from an uploaded BSDL with "
                    f"matching boundary length (saved version predates its "
                    f"parsing)")
        slots.append(_SlotView(position=s["position"], name=name,
                               boundary_length=bl, cells=cells,
                               model_source=source))
    for s in slots:
        if s.cells:
            for row in s.cells:
                if _is_driver(row) and row.get("ccell") is not None and row.get("disval"):
                    s.controls.setdefault(row["ccell"], []).append(row)
    return slots, warnings, ambiguous


def _find_pin(slot: _SlotView, port: str) -> _Pin | None:
    if not slot.cells:
        return None
    canon = _canon_port(port)
    pin = _Pin(slot=slot, port=canon)
    for row in slot.cells:
        if _canon_port(row["port"]) != canon:
            continue
        if _is_driver(row):
            pin.driver_rows.append(row)
        if _is_observer(row):
            pin.observer_rows.append(row)
    if not pin.driver_rows and not pin.observer_rows:
        return None
    return pin


def _driver_descriptor(row: dict, slot: _SlotView, executable: bool) -> dict:
    ccell = row.get("ccell")
    disval = row.get("disval")
    return {
        **_pin_ref(slot, _canon_port(row["port"]), row["cell"]),
        "function": row["function"],
        "tristate": _is_tristate(row),
        "ccell": ccell,
        "disable_value": disval,
        "ccell_chain_bit": (_chain_bit(slot, ccell, executable)
                            if _is_tristate(row) else None),
        "chain_bit": _chain_bit(slot, row["cell"], executable),
    }


def _observer_descriptor(row: dict, slot: _SlotView, executable: bool) -> dict:
    return {
        **_pin_ref(slot, _canon_port(row["port"]), row["cell"]),
        "function": row["function"],
        "chain_bit": _chain_bit(slot, row["cell"], executable),
    }


def _skip(net: _Net, reason: str, detail: str) -> None:
    net.status = "skipped"
    net.skip_reason = reason
    net.skip_detail = detail


def _resolve_net(net: _Net, slots: list[_SlotView], ambiguous: set[str],
                 executable: bool) -> None:
    by_position = {s.position: s for s in slots}
    unresolved: list[str] = []
    for ep in net.raw["endpoints"]:
        pos = ep.get("position")
        if pos is None or pos not in by_position:
            unresolved.append(f"{ep.get('device')!s}.{ep.get('port')!s}: "
                              f"chain position missing or out of range")
            continue
        slot = by_position[pos]
        if slot.name is None:
            _skip(net, SKIP_UNKNOWN_DEVICE,
                  f"chain position {pos} is an unidentified device; "
                  f"endpoint {ep.get('port')!r} cannot be controlled")
            return
        if slot.name in ambiguous:
            _skip(net, SKIP_AMBIGUOUS_DEVICE,
                  f"device {slot.name!r} appears more than once in the chain; "
                  f"endpoints cannot be mapped uniquely")
            return
        if ep.get("device") and ep["device"] != slot.name:
            unresolved.append(
                f"position {pos} holds {slot.name!r}, endpoint names "
                f"{ep['device']!r}")
            continue
        if not slot.cells:
            _skip(net, SKIP_MODEL_UNAVAILABLE,
                  f"device {slot.name!r} (position {pos}) has no parsed "
                  f"BOUNDARY_REGISTER in the saved version or any uploaded "
                  f"BSDL of matching boundary length")
            return
        pin = _find_pin(slot, ep["port"])
        if pin is None:
            unresolved.append(
                f"{slot.name} position {pos} port {ep['port']!r}: no boundary "
                f"cell for that port")
            continue
        net.pins.append(pin)

    if unresolved:
        _skip(net, SKIP_ENDPOINT_NOT_FOUND, "; ".join(unresolved))
        return

    driver_pins = [p for p in net.pins if p.driver_rows]
    if not driver_pins:
        _skip(net, SKIP_NO_DRIVER,
              "no endpoint exposes an output2/output3/bidir boundary cell")
        return

    # Choose a single active driver: prefer a tri-statable pin so the other
    # drivers can be disabled; fall back to an always-on pin only when it is
    # the net's only driver.
    chosen_pin = None
    chosen_row = None
    for p in driver_pins:
        tri = [r for r in p.driver_rows if _is_tristate(r)]
        if tri:
            chosen_pin, chosen_row = p, tri[0]
            break
    if chosen_row is None:
        always_on = [r for p in driver_pins for r in p.driver_rows]
        if len(driver_pins) > 1:
            offenders = sorted({
                f"{p.slot.name}(pos {p.slot.position}).{p.port}"
                for p in driver_pins})
            _skip(net, SKIP_CANNOT_TRISTATE,
                  f"{len(driver_pins)} active drivers and none can be safely "
                  f"tri-stated (output2 / no control cell disable value): "
                  f"{offenders}")
            return
        chosen_pin, chosen_row = driver_pins[0], always_on[0]

    # Every other driver pin must be tri-statable.
    disabled: list[dict] = []
    blockers: list[str] = []
    for p in driver_pins:
        if p is chosen_pin:
            continue
        tri = [r for r in p.driver_rows if _is_tristate(r)]
        if not tri:
            blockers.append(f"{p.slot.name}(pos {p.slot.position}).{p.port} "
                            f"(always-on)")
            continue
        row = tri[0]
        ganged = (row["ccell"] == chosen_row["ccell"]
                  and p.slot.position == chosen_pin.slot.position)
        disabled.append({
            **_driver_descriptor(row, p.slot, executable),
            "held": "hi_z",
            # ganged followers share the driver's enable: their data cell is
            # forced to the driven value so the parallel pin cannot fight
            "ganged": ganged,
        })
    if blockers:
        _skip(net, SKIP_CANNOT_TRISTATE,
              f"single-driver test requires every other driver to enter "
              f"Hi-Z; cannot disable: {blockers}")
        return

    # Observers: every input-capable cell of every endpoint, plus the
    # explicit 1149.1-2001 (IN, <cell>) link of a driven output, if any.
    seen_observers: set[tuple[int, int]] = set()
    observers: list[dict] = []
    for p in net.pins:
        for row in p.observer_rows:
            key = (p.slot.position, row["cell"])
            if key in seen_observers:
                continue
            seen_observers.add(key)
            observers.append(_observer_descriptor(row, p.slot, executable))
        for row in p.driver_rows:
            ref = row.get("input_cell")
            if isinstance(ref, int) and (p.slot.position, ref) not in seen_observers:
                ref_row = next((c for c in (p.slot.cells or [])
                                if c["cell"] == ref), None)
                if ref_row is not None:
                    seen_observers.add((p.slot.position, ref))
                    observers.append(_observer_descriptor(ref_row, p.slot, executable))
    if not observers:
        _skip(net, SKIP_NO_OBSERVER,
              "no input/bidir/observe-only boundary cell on any endpoint")
        return

    net.driver = _driver_descriptor(chosen_row, chosen_pin.slot, executable)
    net.disabled = disabled
    net.forced_active = []
    net.observers = observers


def _control_conflicts(nets: list[_Net], slots: list[_SlotView]) -> None:
    """Block nets whose enabled control cell also releases a pin that is not
    a disabled follower on the same net: either a pin driven by another net
    (shared control across nets) or a pin absent from the board netlist
    (unmanaged output that could fight the selected driver)."""
    by_position = {s.position: s for s in slots}
    planned = [n for n in nets if n.status == "planned"]

    # (position, ccell) -> list of driver rows in the full cell tables
    all_groups: dict[tuple[int, int], list[dict]] = {}
    for s in slots:
        if not s.cells:
            continue
        for ccell, rows in s.controls.items():
            all_groups[(s.position, ccell)] = rows

    # which net claims each physical output pin (if any), and the pin state
    ownership: dict[tuple[int, int], _Net] = {}
    selected_group: dict[_Net, tuple[int, int]] = {}
    for n in planned:
        d = n.driver
        if d["tristate"]:
            selected_group[n] = (d["position"], d["ccell"])
            ownership[(d["position"], d["cell"])] = n
        for f in n.disabled:
            ownership[(f["position"], f["cell"])] = n

    blocked: dict[str, dict[str, list[str]]] = {}

    def block(net: _Net, peer: str, detail: str) -> None:
        entry = blocked.setdefault(net.name, {"peers": [], "details": []})
        if peer not in entry["peers"]:
            entry["peers"].append(peer)
        entry["details"].append(detail)

    for n in planned:
        grp = selected_group.get(n)
        if grp is None:
            continue
        for row in all_groups.get(grp, []):
            key = (grp[0], row["cell"])
            owner = ownership.get(key)
            slot = by_position[grp[0]]
            label = f"{slot.name}(pos {slot.position}).{_canon_port(row['port'])} cell {row['cell']}"
            if owner is n:
                continue  # the selected driver itself or a same-net follower
            if owner is not None:
                other = owner
                block(n, other.name,
                      f"control cell {grp[1]} of {n.driver['device']} is shared "
                      f"with pin {label} driven by net {other.name!r}")
                block(other, n.name,
                      f"control cell {grp[1]} is shared across nets "
                      f"{other.name!r} and {n.name!r} ({label})")
            else:
                block(n, f"unmanaged:{label}",
                      f"enabling control cell {grp[1]} of {n.driver['device']} "
                      f"also releases unmanaged pin {label} not present in the "
                      f"request netlist")

    for n in planned:
        if n.name in blocked:
            info = blocked[n.name]
            _skip(n, SKIP_CONTROL_CONFLICT, "; ".join(sorted(set(info["details"]))))
            n.conflicts_with = sorted(
                p for p in info["peers"] if not p.startswith("unmanaged:"))


# ----------------------------------------------------------------- vectors

def _baseline(slots: list[_SlotView]) -> list[str]:
    """Safe background: control cells at their disable value, everything else
    at its BSDL safe value (X -> 0)."""
    bits: list[str] = []
    for s in reversed(slots):
        if not s.cells:
            continue
        disval_by_cell: dict[int, str] = {}
        for ccell, rows in s.controls.items():
            disvals = sorted({r["disval"] for r in rows if r.get("disval")})
            if disvals:
                disval_by_cell[ccell] = disvals[0]
        for row in sorted(s.cells, key=lambda c: c["cell"]):
            if row["function"] in ("CONTROL", "CONTROLR"):
                bits.append(disval_by_cell.get(
                    row["cell"], row["safe"] if row["safe"] in "01" else "0"))
            else:
                bits.append(row["safe"] if row["safe"] in "01" else "0")
    return bits


def _build_vectors(nets: list[_Net], slots: list[_SlotView],
                   total: int) -> tuple[list[dict], dict[str, str]]:
    """all-0 / per-net walking-1 / all-1. Returns (vectors, code by net)."""
    planned = [n for n in nets if n.status == "planned"]
    p = len(planned)
    order = ["all_zero"] + [f"walk_{n.name}" for n in planned] + ["all_one"]
    driven: dict[str, list[str]] = {n.name: [] for n in planned}
    vectors: list[dict] = []

    for vi, tag in enumerate(order):
        bits = _baseline(slots)
        for n in planned:
            if tag == "all_zero":
                value = "0"
            elif tag == "all_one":
                value = "1"
            else:
                value = "1" if tag == f"walk_{n.name}" else "0"
            driven[n.name].append(value)
            d = n.driver
            bits[d["chain_bit"]] = value
            if d["tristate"]:
                bits[d["ccell_chain_bit"]] = "1" if d["disable_value"] == "0" else "0"
            for f in n.disabled:
                if f.get("ganged"):
                    # shares the driver's enable cell: keep the group enabled
                    # and mirror the driven value on the follower data cell
                    bits[f["chain_bit"]] = value
                else:
                    bits[f["ccell_chain_bit"]] = f["disable_value"]
        vectors.append({"index": vi, "tag": tag,
                        "tdi": "".join(bits), "length": len(bits)})
    codes = {name: "".join(vals) for name, vals in driven.items()}
    return vectors, codes


# ----------------------------------------------------------------- entry: plan

def plan_interconnect(nets: list[dict], version: dict, candidate: int,
                      library: dict[str, list[dict]] | None = None) -> dict:
    """Build the EXTEST interconnect plan.

    ``nets``: [{"name": ..., "endpoints": [{"device", "position", "port"}]}].
    ``version``: row as returned by ``db.get_version``.
    ``library``: device name -> uploaded BSDL model dicts (fallback when the
    version predates BOUNDARY_REGISTER parsing).
    """
    result = version["result"]
    cands = result.get("candidates", [])
    if candidate >= len(cands):
        raise ValueError(
            f"candidate {candidate} not found ({len(cands)} available)")
    chain = cands[candidate]["chain"]
    devices = {d["name"]: d for d in result.get("devices", [])}
    library = library or {}

    slots, warnings, ambiguous = _resolve_slots(chain, devices, library)
    chain_has_unknown = any(s.name is None for s in slots)
    cells_complete = all(s.cells is not None for s in slots if s.name)
    lengths_known = all(s.boundary_length is not None for s in slots)
    fully_describable = (not chain_has_unknown and cells_complete
                         and lengths_known and len(slots) > 0)

    total = 0
    if lengths_known and not chain_has_unknown:
        # offset of a slot = boundary length of every slot TDO-side of it
        for s in slots:
            s.offset = sum(
                t.boundary_length for t in slots if t.position > s.position)
        total = sum(s.boundary_length for s in slots)  # type: ignore[arg-type]

    objs = [_Net(name=n["name"], raw=n) for n in nets]
    for n in objs:
        if n.status == "planned":
            _resolve_net(n, slots, ambiguous, fully_describable)
    _control_conflicts(objs, slots)

    vectors: list[dict] = []
    codes: dict[str, str] = {}
    planned_nets = [n for n in objs if n.status == "planned"]
    if fully_describable and planned_nets:
        vectors, codes = _build_vectors(objs, slots, total)
    elif fully_describable:
        warnings.append("no executable vectors generated: every net was skipped")
    else:
        reasons = []
        if chain_has_unknown:
            reasons.append("chain contains unidentified device slot(s)")
        if not cells_complete:
            reasons.append("BOUNDARY_REGISTER unavailable for some device(s)")
        if not lengths_known:
            reasons.append("BOUNDARY_LENGTH unknown for some device(s)")
        warnings.append("no executable vectors generated: " + "; ".join(reasons))

    instructions = []
    for s in slots:
        model = devices.get(s.name or "")
        opcode = (model or {}).get("opcodes", {}).get("EXTEST")
        instructions.append({
            "position": s.position,
            "device": s.name,
            "extest_opcode": opcode,  # MSB-left as written in BSDL
            "opcode_present": opcode is not None,
        })
    if fully_describable and any(not i["opcode_present"]
                                 for i in instructions if i["device"]):
        missing = [i["device"] for i in instructions
                   if i["device"] and not i["opcode_present"]]
        warnings.append(
            f"EXTEST opcode missing from BSDL opcodes for {missing}; "
            f"IEEE 1149.1 mandates EXTEST but its code cannot be filled in here")

    net_rows = []
    for n in objs:
        net_rows.append({
            "name": n.name,
            "status": n.status,
            "skip_reason": n.skip_reason,
            "skip_detail": n.skip_detail,
            "driver": n.driver,
            "disabled_drivers": n.disabled,
            "forced_active": n.forced_active,
            "observers": n.observers,
            "conflicts_with": n.conflicts_with,
            "code": codes.get(n.name),
        })

    return {
        "version_id": version["id"],
        "candidate": candidate,
        "session": version["session"],
        "executable": fully_describable and bool(planned_nets),
        "boundary_length": total if fully_describable else None,
        "chain_has_unknown": chain_has_unknown,
        "instructions": instructions,
        "chain_segments": [
            {"position": s.position, "device": s.name,
             "boundary_length": s.boundary_length,
             "offset": s.offset if fully_describable else None,
             "cells_available": s.cells is not None,
             "model_source": s.model_source}
            for s in slots
        ],
        "nets": net_rows,
        "vectors": vectors,
        "vector_scheme": ["all_zero", "walking_1", "all_one"] if vectors else [],
        "warnings": warnings,
        "summary": {
            "nets_total": len(objs),
            "nets_planned": sum(1 for n in objs if n.status == "planned"),
            "nets_skipped": sum(1 for n in objs if n.status == "skipped"),
            "vector_count": len(vectors),
            "skip_reasons": _tally(
                n.skip_reason for n in objs if n.status == "skipped"),
        },
    }


def _tally(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


# =============================================================== analysis

def _words_equal(a: str, b: str) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def _flips(word: str) -> list[int]:
    """Vector boundaries where the word changes value."""
    return [i for i in range(1, len(word)) if word[i] != word[i - 1]]


def _bridge_match(observed: str, own: str, other: str) -> dict | None:
    """How observed pin behavior follows another net's driven word."""
    if len(observed) != len(other):
        return None
    inv = "".join("1" if b == "0" else "0" for b in other)
    differs_own = observed != own
    if not differs_own:
        return None
    if observed == other:
        return {"kind": "short_equal", "strength": 5, "polarity": "same"}
    if observed == inv:
        return {"kind": "short_inverted", "strength": 4, "polarity": "inverted"}
    and_w = "".join("1" if a == "1" and b == "1" else "0"
                    for a, b in zip(own, other))
    or_w = "".join("1" if a == "1" or b == "1" else "0"
                   for a, b in zip(own, other))
    if observed == and_w and and_w != own:
        return {"kind": "wired_and", "strength": 3, "polarity": "same"}
    if observed == or_w and or_w != own:
        return {"kind": "wired_or", "strength": 3, "polarity": "same"}
    # synchronous toggle: same flip boundaries as the other net's word
    # (possibly inverted), even though levels don't line up exactly.
    if _flips(observed) == _flips(other) and set(observed) != {observed[0]}:
        return {"kind": "sync_toggle", "strength": 2,
                "polarity": ("inverted"
                             if _flips(observed) == _flips(inv)
                             and observed != other else "same")}
    return None


def analyze_interconnect(plan: dict, measurements: list[dict]) -> dict:
    """Unload measured TDO scans of a saved plan and diagnose interconnect.

    ``measurements``: [{"vector_index": int, "tdo": str, "label"?: str}].
    Raises ValueError for malformed input or a non-executable plan.
    """
    if not plan.get("executable"):
        raise ValueError(
            "plan is not executable (unknown devices / missing BSDL models); "
            "measured TDO cannot be mapped to pins")
    L = plan["boundary_length"]
    vecs = sorted(plan["vectors"], key=lambda v: v["index"])
    expected_indexes = [v["index"] for v in vecs]
    got = {m["vector_index"]: m for m in measurements}
    missing = [i for i in expected_indexes if i not in got]
    if missing:
        raise ValueError(f"missing TDO measurements for vectors {missing}")
    dup = len(measurements) - len(got)
    if dup:
        raise ValueError("duplicate vector_index values in measurements")

    captures: dict[int, str] = {}
    vector_warnings: list[dict] = []
    tdi_by_index = {v["index"]: v["tdi"] for v in vecs}
    for i in expected_indexes:
        m = got[i]
        tdo = m["tdo"]
        if len(tdo) < L:
            raise ValueError(
                f"vector {i}: TDO has {len(tdo)} bits, need {L} chain bits")
        captures[i] = tdo[:L]
        if len(tdo) > L:
            tdi = tdi_by_index[i]
            tail, probe = tdo[L:], tdi[: len(tdo) - L]
            n = min(16, len(tail), len(probe))
            if n >= 4 and tail[:n] != probe[:n]:
                vector_warnings.append({
                    "vector_index": i,
                    "kind": "echo_mismatch",
                    "detail": f"TDO tail after {L} capture bits does not echo "
                              f"the shifted TDI; capture alignment uncertain",
                })

    order = expected_indexes
    nets = [n for n in plan["nets"] if n["status"] == "planned"]
    pin_reports_by_net: dict[str, list[dict]] = {}
    findings: list[dict] = []
    driven_by_net = {n["name"]: n["code"] for n in nets}

    for n in nets:
        own = n["code"]
        pin_reports = []
        for obs in n["observers"]:
            cb = obs["chain_bit"]
            word = "".join(captures[i][cb] for i in order)
            mismatch_vectors = [i for i, got_b, exp_b
                                in zip(order, word, own) if got_b != exp_b]

            best = None
            for other in nets:
                if other["name"] == n["name"]:
                    continue
                match = _bridge_match(word, own, other["code"])
                if match and (best is None or match["strength"] > best["strength"]):
                    best = {**match, "other_net": other["name"],
                            "other_driver": _loc(other["driver"])}
            if word == own:
                verdict = "pass"
            elif set(word) == {"0"} or set(word) == {"1"}:
                verdict = "fixed_high" if word[0] == "1" else "fixed_low"
            elif best is not None:
                verdict = "bridge"
            else:
                verdict = "unexpected_response"

            report = {
                "device": obs["device"],
                "position": obs["position"],
                "port": obs["port"],
                "cell": obs["cell"],
                "chain_bit": cb,
                "observed_word": word,
                "expected_word": own,
                "mismatch_vectors": mismatch_vectors,
                "verdict": verdict,
                "bridge": best,
            }
            pin_reports.append(report)
        pin_reports_by_net[n["name"]] = pin_reports

        # ---- net-level verdict -------------------------------------------
        verdicts = {p["verdict"] for p in pin_reports}
        net_verdict, net_detail = _net_verdict(own, pin_reports)

        for p in pin_reports:
            if p["verdict"] != "pass":
                findings.append(_finding(n, p, net_verdict, driven_by_net))

        n["pin_results"] = pin_reports  # temporary, stripped below
        n["analysis_verdict"] = net_verdict
        n["analysis_detail"] = net_detail
        n["verdicts_seen"] = sorted(verdicts)

    # de-duplicate bridge candidates (both sides may point at each other)
    bridges = [f for f in findings if f["kind"] in (
        "bridge_short", "bridge_sync_toggle")]
    seen_bridge = set()
    bridge_rows = []
    for f in bridges:
        key = tuple(sorted((f["net"], f["bridge_with"]))), f["bridge_kind"]
        if key in seen_bridge:
            continue
        seen_bridge.add(key)
        bridge_rows.append({
            "kind": f["bridge_kind"],
            "nets": sorted((f["net"], f["bridge_with"])),
            "polarity": f.get("polarity"),
            "witness": {
                "net": f["net"],
                "device": f["device"],
                "position": f["position"],
                "port": f["port"],
                "cell": f["cell"],
                "chain_bit": f["chain_bit"],
                "vectors": f["vectors"],
                "observed_word": f["observed_word"],
                "expected_word": f["expected_word"],
            },
            "detail": f["detail"],
        })

    net_rows = []
    counts = {"pass": 0, "suspected_open": 0, "fixed_level": 0,
              "bridge": 0, "unexpected_response": 0}
    for n in nets:
        pins = n.pop("pin_results")
        net_rows.append({
            "name": n["name"],
            "driver": n["driver"],
            "observers": n["observers"],
            "code": n["code"],
            "verdict": n["analysis_verdict"],
            "detail": n["analysis_detail"],
            "pin_verdicts": n["verdicts_seen"],
            "pins": pins,
        })
        top = n["analysis_verdict"]
        counts[top if top in counts else "unexpected_response"] += 1

    skipped = [{"name": n["name"], "skip_reason": n["skip_reason"],
                "skip_detail": n["skip_detail"]}
               for n in plan["nets"] if n["status"] == "skipped"]

    overall = "pass" if counts["pass"] == len(nets) else "fail"
    return {
        "plan_id": plan.get("plan_id"),
        "version_id": plan["version_id"],
        "candidate": plan["candidate"],
        "session": plan["session"],
        "boundary_length": L,
        "vector_order": order,
        "measurements": [
            {"vector_index": m["vector_index"],
             "label": m.get("label"),
             "tdo_length": len(m["tdo"]),
             "capture_bits_used": L}
            for m in (got[i] for i in order)
        ],
        "nets": net_rows,
        "skipped_nets": skipped,
        "findings": findings,
        "bridge_candidates": bridge_rows,
        "warnings": vector_warnings,
        "summary": {
            "overall": overall,
            "nets_total": len(nets),
            **counts,
            "nets_skipped_in_plan": len(skipped),
            "bridge_candidates": len(bridge_rows),
            "findings": len(findings),
        },
    }


def _net_verdict(own: str, pins: list[dict]) -> tuple[str, str]:
    bad = [p for p in pins if p["verdict"] != "pass"]
    if not bad:
        return "pass", "all observer pins returned the driven word"
    if all(p["verdict"] == "bridge" for p in bad) and not any(
            p["verdict"] not in ("pass", "bridge") for p in pins):
        names = sorted({p["bridge"]["other_net"] for p in bad})
        return "bridge", f"observer follows net(s) {names}"
    constants = [p for p in bad
                 if p["verdict"] in ("fixed_low", "fixed_high")]
    bridges = [p for p in bad if p["verdict"] == "bridge"]
    values = {p["observed_word"][0] for p in constants}
    driven_both = "0" in own and "1" in own
    if bridges:
        names = sorted({p["bridge"]["other_net"] for p in bridges})
        return "bridge", (f"{len(bridges)} pin(s) follow other net(s) {names}; "
                          f"other pins {[p['verdict'] for p in bad if p['verdict'] != 'bridge']}")
    if len(constants) == len(bad):
        if len(values) > 1:
            return ("suspected_open",
                    "observer pins hold conflicting constant levels despite a "
                    "common net: open trace / local stuck pins")
        level = constants[0]["observed_word"][0]
        if len(pins) == 1 and driven_both:
            return ("suspected_open",
                    f"sole observer reads constant {level} while the net was "
                    f"driven 0 and 1: open connection with pull or a local "
                    f"stuck-at-{level} (cannot be distinguished here)")
        label = f"fixed level {level}"
        if driven_both:
            return ("fixed_level",
                    f"every observer reads {label} although the driver "
                    f"toggled: net stuck-at-{level}")
        return ("fixed_level", f"all observers read {label}")
    return ("suspected_open",
            "some observers stopped following the driver while others did: "
            "open trace between driver and the constant/dead pins")


def _loc(driver: dict) -> dict:
    return {"device": driver["device"], "position": driver["position"],
            "port": driver["port"], "cell": driver["cell"],
            "chain_bit": driver.get("chain_bit")}


def _finding(net: dict, pin: dict, net_verdict: str,
             driven_by_net: dict[str, str]) -> dict:
    v = pin["verdict"]
    if v == "bridge":
        b = pin["bridge"]
        kind = ("bridge_sync_toggle" if b["kind"] == "sync_toggle"
                else "bridge_short")
        detail = (f"pin {pin['device']}.{pin['port']} on net {net['name']!r} "
                  f"returns the {b['kind'].replace('_', ' ')} signature of "
                  f"net {b['other_net']!r} ({b['polarity']})")
        return {
            "kind": kind,
            "net": net["name"],
            "bridge_with": b["other_net"],
            "bridge_kind": b["kind"],
            "polarity": b["polarity"],
            "device": pin["device"],
            "position": pin["position"],
            "port": pin["port"],
            "cell": pin["cell"],
            "chain_bit": pin["chain_bit"],
            "vectors": pin["mismatch_vectors"],
            "observed_word": pin["observed_word"],
            "expected_word": pin["expected_word"],
            "other_driven_word": driven_by_net.get(b["other_net"]),
            "detail": detail,
        }
    if v in ("fixed_low", "fixed_high"):
        level = v[-1]
        kind = "fixed_level" if net_verdict == "fixed_level" else "suspected_open"
        detail = (f"pin {pin['device']}.{pin['port']} reads constant {level} "
                  f"on vectors {pin['mismatch_vectors']} where {net['name']!r} "
                  f"was driven to the opposite level")
        return {
            "kind": kind,
            "fixed_level": level,
            "net": net["name"],
            "device": pin["device"],
            "position": pin["position"],
            "port": pin["port"],
            "cell": pin["cell"],
            "chain_bit": pin["chain_bit"],
            "vectors": pin["mismatch_vectors"],
            "observed_word": pin["observed_word"],
            "expected_word": pin["expected_word"],
            "detail": detail,
        }
    return {
        "kind": "unexpected_response",
        "net": net["name"],
        "device": pin["device"],
        "position": pin["position"],
        "port": pin["port"],
        "cell": pin["cell"],
        "chain_bit": pin["chain_bit"],
        "vectors": pin["mismatch_vectors"],
        "observed_word": pin["observed_word"],
        "expected_word": pin["expected_word"],
        "detail": (f"pin {pin['device']}.{pin['port']} response does not match "
                   f"the driven word or any other single-net signature"),
    }
