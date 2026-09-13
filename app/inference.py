"""JTAG scan-chain inference engine.

Chain convention: position 0 is closest to TDI, position n-1 drives TDO.
Bit streams are in wire order (index 0 = first bit out on TDO). During a
scan the first L bits out of TDO are the registers' captured content
(L = total register length); after that, TDO echoes TDI delayed by L.
"""
from __future__ import annotations

from dataclasses import dataclass, field

MAX_CANDIDATES = 2000
TOP_N = 10


@dataclass
class Device:
    name: str
    ir_length: int
    ir_capture: str  # shift-out order, may contain 'X'
    opcodes: dict[str, str] = field(default_factory=dict)  # MSB-left, as in BSDL
    idcode_value: int | None = None
    idcode_mask: int | None = None
    boundary_length: int | None = None
    boundary_cells: list[dict] | None = None
    count: int = 1


@dataclass
class Slot:
    device: Device | None  # None = unidentified device
    ir_length: int
    capture_observed: str
    idcode_observed: int | None = None
    idcode_match: bool | None = None
    idcode_mismatch_bits: list[int] = field(default_factory=list)


@dataclass
class Candidate:
    slots_tdo: list[Slot]  # TDO-side first
    ir_offset: int
    score: int = 0
    constraints: list[str] = field(default_factory=list)
    unexplained: list[str] = field(default_factory=list)

    @property
    def chain(self) -> list[Slot]:  # TDI-side first
        return list(reversed(self.slots_tdo))

    def signature(self) -> tuple:
        return tuple((s.device.name if s.device else "?", s.ir_length) for s in self.slots_tdo)


# ---------------------------------------------------------------- matching

def _match(pattern: str, observed: str, unrel: set[int]) -> bool:
    """pattern/observed in shift-out order; unrel = don't-care indices."""
    if len(observed) < len(pattern):
        return False
    for i, p in enumerate(pattern):
        if p == "X" or i in unrel:
            continue
        if observed[i] != p:
            return False
    return True


def find_offsets(tdi: str, tdo: str, unrel: set[int]) -> list[int]:
    """All positions where the TDI probe reappears in TDO (= register length).

    The echo may be truncated at the end of the captured stream, so a
    partial tail match is accepted as long as at least 8 bits are compared.
    """
    if not tdi or not tdo:
        return []
    probe = tdi[:32]
    min_bits = min(8, len(probe))
    offs = []
    for off in range(len(tdo)):
        compared = 0
        ok = True
        for i, b in enumerate(probe):
            pos = off + i
            if pos >= len(tdo):
                break
            if pos in unrel:
                continue
            compared += 1
            if tdo[pos] != b:
                ok = False
                break
        if ok and compared >= min_bits:
            offs.append(off)
    return offs


def segment(capture: str, devices: list[Device], unrel: set[int],
            max_unknown: int, limit: int = MAX_CANDIDATES) -> list[list[Slot]]:
    """Split the IR capture region (TDO-side first) into device IR segments
    whose INSTRUCTION_CAPTURE patterns match. Unknown devices are allowed as
    length-2..16 segments showing the mandatory '01' LSB capture pair."""
    results: list[list[Slot]] = []
    counts = [d.count for d in devices]

    def rec(pos: int, slots: list[Slot], unknown_used: int) -> None:
        if len(results) >= limit:
            return
        if pos == len(capture):
            results.append(list(slots))
            return
        for i, d in enumerate(devices):
            if counts[i] <= 0 or pos + d.ir_length > len(capture):
                continue
            seg = capture[pos:pos + d.ir_length]
            lu = {k - pos for k in unrel if pos <= k < pos + d.ir_length}
            if _match(d.ir_capture, seg, lu):
                counts[i] -= 1
                slots.append(Slot(d, d.ir_length, seg))
                rec(pos + d.ir_length, slots, unknown_used)
                slots.pop()
                counts[i] += 1
        if unknown_used < max_unknown:
            for length in range(2, 17):
                if pos + length > len(capture):
                    break
                seg = capture[pos:pos + length]
                lu = {k - pos for k in unrel if pos <= k < pos + length}
                # IEEE 1149.1: IR capture LSBs are always "01"
                if all(k in lu or seg[k] == want for k, want in ((0, "1"), (1, "0"))):
                    slots.append(Slot(None, length, seg))
                    rec(pos + length, slots, unknown_used + 1)
                    slots.pop()

    rec(0, [], 0)
    return results


# ---------------------------------------------------------------- evaluation

def _validate_extra_ir(cand: Candidate, ci: int, cap, unrel: set[int], offset: int) -> None:
    offs = find_offsets(cap.tdi, cap.tdo, unrel)
    if offset not in offs:
        cand.unexplained.append(
            f"IR capture #{ci}: alignment {offset} not found (possible offsets {offs[:8]})")
        return
    pos, ok = 0, True
    for s in cand.slots_tdo:
        if s.device:
            seg = cap.tdo[pos:pos + s.ir_length]
            lu = {k - pos for k in unrel if pos <= k < pos + s.ir_length}
            if not _match(s.device.ir_capture, seg, lu):
                ok = False
        pos += s.ir_length
    if ok:
        cand.score += 1
        cand.constraints.append(f"IR capture #{ci}: consistent at offset {offset}")
    else:
        cand.unexplained.append(f"IR capture #{ci}: capture-region mismatch at offset {offset}")


def _eval_dr(cand: Candidate, cap, unrel: set[int], bit_order: str, ci: int) -> int:
    inst = (cap.instruction or "").upper()
    score = 0
    if inst == "BYPASS":
        offs = find_offsets(cap.tdi, cap.tdo, unrel)
        n = len(cand.slots_tdo)
        if offs:
            if n in offs:
                score += 2
                cand.constraints.append(f"BYPASS scan: device count {n} confirmed")
            else:
                score -= 2
                cand.unexplained.append(
                    f"BYPASS scan implies {offs} device(s), candidate has {n}")
        for i in range(min(n, len(cap.tdo))):
            if i not in unrel and cap.tdo[i] != "0":
                cand.unexplained.append(f"BYPASS capture bit {i} is 1 (spec expects 0)")
    elif inst in ("IDCODE", "DEVICE_ID"):
        pos = 0
        for s in cand.slots_tdo:
            if s.device and s.device.idcode_value is not None:
                chunk = cap.tdo[pos:pos + 32]
                if len(chunk) < 32:
                    cand.unexplained.append("IDCODE scan: TDO stream truncated")
                    break
                obs = int(chunk, 2) if bit_order == "msb_first" else int(chunk[::-1], 2)
                mask = s.device.idcode_mask or 0xFFFFFFFF
                for k in range(32):  # unreliable bits are excluded from the mask
                    if pos + k in unrel:
                        bit_index = (31 - k) if bit_order == "msb_first" else k
                        mask &= ~(1 << bit_index)
                diff = (obs ^ s.device.idcode_value) & mask
                s.idcode_observed = obs
                if diff == 0:
                    s.idcode_match = True
                    score += 2
                    cand.constraints.append(
                        f"IDCODE match: {s.device.name} = 0x{obs:08X}")
                else:
                    s.idcode_match = False
                    s.idcode_mismatch_bits = [b for b in range(32) if diff >> b & 1]
                    score -= 2
                    cand.unexplained.append(
                        f"IDCODE mismatch: {s.device.name} observed 0x{obs:08X} "
                        f"expected 0x{s.device.idcode_value:08X} "
                        f"(mask 0x{mask:08X}, bits {s.idcode_mismatch_bits})")
                pos += 32
            else:  # device without IDCODE sits in BYPASS
                if pos < len(cap.tdo) and pos not in unrel:
                    if cap.tdo[pos] == "0":
                        score += 1
                    else:
                        score -= 1
                        cand.unexplained.append(
                            f"IDCODE scan: bypass bit at TDO {pos} is 1 (expected 0)")
                pos += 1
    elif inst in ("SAMPLE", "PRELOAD"):
        offs = find_offsets(cap.tdi, cap.tdo, unrel)
        known = [s.device.boundary_length for s in cand.slots_tdo]
        if offs and all(b is not None for b in known):
            total = sum(known)
            if total in offs:
                score += 2
                cand.constraints.append(
                    f"{inst} scan: boundary length {total} confirmed")
            else:
                score -= 2
                cand.unexplained.append(
                    f"{inst} scan: BOUNDARY_LENGTH sum {total} != observed DR length {offs[:8]}")
    return score


def _locks_ok(cand: Candidate, locks) -> bool:
    chain = cand.chain
    for lk in locks:
        if lk.position >= len(chain):
            return False
        slot = chain[lk.position]
        name = slot.device.name if slot.device else "unknown"
        if name != lk.device:
            return False
    return True


# ---------------------------------------------------------------- main entry

def infer(devices: list[Device], captures, unrel_map: dict[int, set[int]],
          bit_order: str, locks, max_unknown: int):
    ir_caps = [(i, c) for i, c in enumerate(captures) if c.kind == "ir"]
    candidates: list[Candidate] = []
    notes: list[str] = []

    if ir_caps:
        i0, c0 = ir_caps[0]
        u0 = unrel_map.get(i0, set())
        offsets = find_offsets(c0.tdi, c0.tdo, u0)[:50]
        if not offsets:
            notes.append("IR capture #0: TDI pattern not found in TDO "
                         "(alignment failed; check wiring or mark unreliable bits)")
        seen = set()
        for off in offsets:
            for slots in segment(c0.tdo[:off], devices, u0, max_unknown):
                cand = Candidate(slots_tdo=slots, ir_offset=off)
                cand.score += 2 * sum(1 for s in slots if s.device)
                cand.constraints.append(
                    f"IR capture #0: {off}-bit capture region -> {len(slots)} device(s)")
                for s in slots:
                    if s.device is None:
                        cand.unexplained.append(
                            f"unidentified device: IR length {s.ir_length}, "
                            f"capture {s.capture_observed}")
                for ci, cap in ir_caps[1:]:
                    _validate_extra_ir(cand, ci, cap, unrel_map.get(ci, set()), off)
                for ci, cap in enumerate(captures):
                    if cap.kind == "dr":
                        cand.score += _eval_dr(cand, cap, unrel_map.get(ci, set()), bit_order, ci)
                if cand.signature() not in seen:
                    seen.add(cand.signature())
                    candidates.append(cand)
    else:
        notes.append("no IR capture supplied: chain structure cannot be segmented")

    if locks:
        candidates = [c for c in candidates if _locks_ok(c, locks)]
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:TOP_N]

    checks = run_checks(devices, captures, unrel_map, candidates, locks, max_unknown)
    if notes:
        checks["notes"] = notes
    return candidates, checks


# ---------------------------------------------------------------- diagnostics

def _best_ir_quality(tdo: str, tdi: str, devices: list[Device],
                     unrel: set[int], max_unknown: int) -> int | None:
    best = None
    for off in find_offsets(tdi, tdo, unrel)[:50]:
        for slots in segment(tdo[:off], devices, unrel, max_unknown, limit=200):
            q = (2 * sum(1 for s in slots if s.device)
                 - sum(1 for s in slots if s.device is None))
            best = q if best is None else max(best, q)
    return best


def run_checks(devices, captures, unrel_map, candidates, locks, max_unknown) -> dict:
    checks: dict = {}

    const = [i for i, c in enumerate(captures) if c.tdo and len(set(c.tdo)) == 1]
    checks["tdo_constant"] = {
        "flag": bool(const),
        "captures": const,
        "detail": ("TDO never toggles: open chain, shorted TDO or TAP held in reset"
                   if const else ""),
    }

    offby = []
    for i, c in enumerate(captures):
        if c.kind != "ir":
            continue
        q0 = _best_ir_quality(c.tdo, c.tdi, devices, unrel_map.get(i, set()), max_unknown)
        shifted = [q for q in (
            _best_ir_quality(c.tdo[1:], c.tdi, devices, set(), max_unknown),
            _best_ir_quality("0" + c.tdo, c.tdi, devices, set(), max_unknown),
            _best_ir_quality("1" + c.tdo, c.tdi, devices, set(), max_unknown),
        ) if q is not None]
        if shifted and (q0 is None or max(shifted) > q0):
            offby.append(i)
    checks["overall_offset_by_one"] = {
        "flag": bool(offby),
        "captures": offby,
        "detail": ("TDO aligns better when shifted by one bit: clock/probe skew suspected"
                   if offby else ""),
    }

    conflicts: list[str] = []
    ir_offsets: list[int] = []
    for i, c in enumerate(captures):
        if c.kind == "ir":
            ir_offsets.extend(find_offsets(c.tdi, c.tdo, unrel_map.get(i, set())))
    locked_sum = 0
    for lk in locks:
        d = next((d for d in devices if d.name == lk.device), None)
        if d:
            locked_sum += d.ir_length
    if locks and ir_offsets and locked_sum > max(ir_offsets):
        conflicts.append(
            f"locked devices need {locked_sum} IR bits but observed IR chain "
            f"is at most {max(ir_offsets)} bits")
    if candidates:
        best = candidates[0]
        for i, c in enumerate(captures):
            if c.kind == "dr" and (c.instruction or "").upper() in ("SAMPLE", "PRELOAD"):
                offs = find_offsets(c.tdi, c.tdo, unrel_map.get(i, set()))
                blens = [s.device.boundary_length for s in best.slots_tdo if s.device]
                if offs and blens and all(b is not None for b in blens):
                    total = sum(blens)
                    if total not in offs:
                        conflicts.append(
                            f"capture #{i}: BOUNDARY_LENGTH sum {total} != "
                            f"observed DR length {offs[:8]}")
    checks["bsdl_length_conflict"] = {"flag": bool(conflicts), "detail": conflicts}

    mism = []
    if candidates:
        for pos, s in enumerate(candidates[0].slots_tdo):
            if s.device and s.idcode_match is False:
                hint = ""
                if s.idcode_mismatch_bits and all(b >= 28 for b in s.idcode_mismatch_bits):
                    hint = "mismatch confined to version field; consider mask 0x0FFFFFFF"
                mism.append({
                    "tdo_position": pos,
                    "device": s.device.name,
                    "mismatch_bits": s.idcode_mismatch_bits,
                    "observed": hex(s.idcode_observed or 0),
                    "expected": hex(s.device.idcode_value or 0),
                    "mask": hex(s.device.idcode_mask or 0),
                    "hint": hint,
                })
    checks["idcode_mask_mismatch"] = {"flag": bool(mism), "detail": mism}

    tied: list[Candidate] = []
    if candidates:
        top = candidates[0].score
        tied = [c for c in candidates if c.score == top]
    checks["indistinguishable_candidates"] = {
        "flag": len(tied) > 1,
        "count": len(tied),
        "detail": ([["%s(ir=%d)" % (n, l) for n, l in c.signature()] for c in tied]
                   if len(tied) > 1 else []),
    }
    return checks


# ---------------------------------------------------------------- serialization

def candidate_to_dict(cand: Candidate) -> dict:
    return {
        "score": cand.score,
        "ir_offset": cand.ir_offset,
        "chain": [
            {
                "position": pos,
                "device": s.device.name if s.device else None,
                "status": "identified" if s.device else "unknown",
                "ir_length": s.ir_length,
                "ir_capture_observed": s.capture_observed,
                "idcode_observed": hex(s.idcode_observed) if s.idcode_observed is not None else None,
                "idcode_match": s.idcode_match,
            }
            for pos, s in enumerate(cand.chain)
        ],
        "constraints": cand.constraints,
        "unexplained": cand.unexplained,
    }
