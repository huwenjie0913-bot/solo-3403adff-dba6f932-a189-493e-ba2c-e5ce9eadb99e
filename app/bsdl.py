"""Minimal BSDL (IEEE 1149.1) parser.

Extracts what chain reconstruction and board interconnect testing need:
  - entity name
  - INSTRUCTION_LENGTH
  - INSTRUCTION_OPCODE  (IDCODE / BYPASS / SAMPLE / PRELOAD / EXTEST / ...)
  - INSTRUCTION_CAPTURE (may contain X don't-care bits)
  - IDCODE_REGISTER     (32-bit value, X bits become mask 0)
  - BOUNDARY_LENGTH
  - BOUNDARY_REGISTER   (per-cell number, port, function, safe value,
                         control cell / disable value / disable result and
                         the optional 1149.1-2001 ``(IN, <cell>)`` input link)

Bit-string convention used across the whole project:
  * BSDL writes patterns MSB-left ("0001" means bit3..bit0 = 0,0,0,1).
  * The inference engine works in *shift-out order*: index 0 is the first
    bit seen on TDO, which for a register shifted LSB-first is bit 0.
  * `parse_bsdl` returns patterns in BSDL order; conversion to shift order
    happens in `app.models.device_from_in` (a plain reverse).

Boundary cell numbers are used verbatim: per IEEE 1149.1 cell 0 is the
boundary-register cell closest to TDO, so within one device the shift-order
bit index of a cell equals its BSDL cell number.
"""
from __future__ import annotations

import re


class BsdlError(ValueError):
    pass


# Canonical boundary-cell functions (BSDL spelling, underscore form).
CELL_FUNCTIONS = {
    "INPUT", "OUTPUT2", "OUTPUT3", "BIDIR", "INTERNAL",
    "CONTROL", "CONTROLR", "OBSERVE_ONLY", "CLOCK",
}
# Very old BSDL files spell output2 as plain "output".
FUNCTION_ALIASES = {"OUTPUT": "OUTPUT2"}


def _attr(text: str, name: str) -> str | None:
    m = re.search(
        r"attribute\s+" + re.escape(name) + r"\s+of\s+\w+\s*:\s*entity\s+is\s+(.*?);",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1) if m else None


# ----------------------------------------------------------- BOUNDARY_REGISTER

def _split_top_level(inner: str) -> list[str]:
    """Split a tuple body on commas at parenthesis depth 0."""
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i])
            start = i + 1
    parts.append(inner[start:])
    return [p.strip() for p in parts]


def _iter_tuples(body: str):
    """Yield (cell_number, inner text) of ``num (...)`` boundary-cell tuples,
    honoring nested parentheses (the 1149.1-2001 input specification)."""
    for m in re.finditer(r"(?<![\w.])(\d+)\s*\(", body):
        open_at = body.index("(", m.start())
        depth = 0
        end = open_at
        while end < len(body):
            if body[end] == "(":
                depth += 1
            elif body[end] == ")":
                depth -= 1
                if depth == 0:
                    break
            end += 1
        if end < len(body):
            yield int(m.group(1)), body[open_at + 1:end]


def _norm_function(token: str) -> str:
    name = re.sub(r"\s+", "_", token.strip().upper())
    name = FUNCTION_ALIASES.get(name, name)
    if name not in CELL_FUNCTIONS:
        raise BsdlError(f"unknown boundary-cell function {token.strip()!r}")
    return name


def _capabilities(function: str, ccell: int | None, disval: str | None) -> list[str]:
    """Derived roles of a cell. ``tristate`` requires a usable control cell
    with a known disable value; without it an output3/bidir driver is an
    always-on driver as far as safe vector planning is concerned."""
    caps: list[str] = []
    if function in ("INPUT", "CLOCK", "OBSERVE_ONLY"):
        caps.append("input")
    elif function in ("OUTPUT2", "OUTPUT3", "BIDIR"):
        caps.append("output")
        if function == "BIDIR":
            caps.append("input")
        if function in ("OUTPUT3", "BIDIR") and ccell is not None and disval in ("0", "1"):
            caps.append("tristate")
    elif function in ("CONTROL", "CONTROLR"):
        caps.append("control")
    return caps


def _parse_boundary_register(raw: str) -> list[dict]:
    # Drop VHDL comments, keep only the quoted string fragments.
    cleaned = re.sub(r"--[^\n]*", " ", raw)
    fragments = re.findall(r'"([^"]*)"', cleaned)
    body = " ".join(fragments) if fragments else cleaned.replace('"', " ")
    body = body.replace("&", " ")

    cells: list[dict] = []
    for number, inner in _iter_tuples(body):
        parts = _split_top_level(inner)
        if len(parts) < 4:
            raise BsdlError(
                f"boundary cell {number}: expected at least 4 fields "
                f"(cell type, port, function, safe), got {len(parts)}")
        cell_type, port, func_tok, safe_tok, *tail = parts

        port = re.sub(r"\s+", "", port)
        if port not in ("*", "") and not re.fullmatch(r"[A-Za-z_][\w]*(?:[(\[].+?[)\]])?", port):
            raise BsdlError(f"boundary cell {number}: invalid port identifier {port!r}")
        if port == "":
            port = "*"
        function = _norm_function(func_tok)
        safe_m = re.search(r"[01xX]", safe_tok)
        if not safe_m:
            raise BsdlError(f"boundary cell {number}: invalid safe value {safe_tok!r}")
        safe = safe_m.group(0).upper()

        # Optional 1149.1-2001 input specification: (IN, <cell>[, <name>]).
        input_cell = None
        if tail and tail[-1].strip().startswith("("):
            spec = tail.pop().strip()
            m_in = re.search(r"\bIN\s*,([^,)]+)", spec, re.IGNORECASE)
            if m_in:
                ref = m_in.group(1).strip()
                if ref.isdigit():
                    input_cell = int(ref)
                # named cell references are left unresolved; planners then
                # fall back to the data cell's own capture.

        ccell = None
        disval = None
        disable_result = None
        if tail:
            if not tail[0].strip().isdigit():
                raise BsdlError(
                    f"boundary cell {number}: control cell must be a number, "
                    f"got {tail[0].strip()!r}")
            ccell = int(tail[0].strip())
            if len(tail) >= 2:
                dv = re.sub(r"[\s()]", "", tail[1]).upper()
                if dv not in ("0", "1"):
                    raise BsdlError(
                        f"boundary cell {number}: disable value must be 0/1, "
                        f"got {tail[1].strip()!r}")
                disval = dv
            if len(tail) >= 3:
                disable_result = re.sub(r"[\s()]", "", tail[2]).upper() or None

        cells.append({
            "cell": number,
            "port": port,
            "cell_type": cell_type.strip(),
            "function": function,
            "safe": safe,
            "ccell": ccell,
            "disval": disval,
            "disable_result": disable_result,
            "input_cell": input_cell,
            "capabilities": _capabilities(function, ccell, disval),
        })

    if not cells:
        raise BsdlError("BOUNDARY_REGISTER attribute contains no cell tuples")

    seen = set()
    for c in cells:
        if c["cell"] in seen:
            raise BsdlError(f"duplicate boundary cell number {c['cell']}")
        seen.add(c["cell"])
    cells.sort(key=lambda c: c["cell"])
    if [c["cell"] for c in cells] != list(range(len(cells))):
        raise BsdlError(
            f"boundary cell numbers must be contiguous 0..{len(cells) - 1}, "
            f"got {sorted(seen)}")
    return cells


def parse_bsdl(text: str) -> dict:
    entity = re.search(r"entity\s+(\w+)\s+is", text, re.IGNORECASE)
    if not entity:
        raise BsdlError("no 'entity <name> is' declaration found")
    name = entity.group(1)

    ir_len = _attr(text, "INSTRUCTION_LENGTH")
    if not ir_len:
        raise BsdlError("INSTRUCTION_LENGTH attribute not found")
    m = re.search(r"(\d+)", ir_len)
    ir_length = int(m.group(1))

    opcodes: dict[str, str] = {}
    raw_ops = _attr(text, "INSTRUCTION_OPCODE")
    if raw_ops:
        for op_name, bits in re.findall(r"(\w+)\s*\(\s*([01xX]+)\s*\)", raw_ops):
            opcodes[op_name.upper()] = bits
    if not opcodes:
        raise BsdlError("INSTRUCTION_OPCODE attribute not found")
    for op_name, bits in opcodes.items():
        if len(bits) != ir_length:
            raise BsdlError(
                f"opcode {op_name} has {len(bits)} bits, expected {ir_length}"
            )

    ir_capture = ""
    raw_cap = _attr(text, "INSTRUCTION_CAPTURE")
    if raw_cap:
        m = re.search(r'"\s*([01xX]+)\s*"', raw_cap)
        if m:
            ir_capture = m.group(1).upper()
            if len(ir_capture) != ir_length:
                raise BsdlError(
                    f"INSTRUCTION_CAPTURE has {len(ir_capture)} bits, expected {ir_length}"
                )

    idcode_value = None
    idcode_mask = None
    raw_id = _attr(text, "IDCODE_REGISTER")
    if raw_id:
        fields = re.findall(r'"\s*([01xX]+)\s*"', raw_id)
        bits = "".join(fields).upper()
        if len(bits) != 32:
            raise BsdlError(f"IDCODE_REGISTER has {len(bits)} bits, expected 32")
        idcode_value = int(bits.replace("X", "0"), 2)
        idcode_mask = int("".join("0" if b == "X" else "1" for b in bits), 2)

    boundary_length = None
    raw_bl = _attr(text, "BOUNDARY_LENGTH")
    if raw_bl:
        m = re.search(r"(\d+)", raw_bl)
        if m:
            boundary_length = int(m.group(1))

    boundary_cells: list[dict] | None = None
    raw_br = _attr(text, "BOUNDARY_REGISTER")
    if raw_br:
        boundary_cells = _parse_boundary_register(raw_br)
        if boundary_length is None:
            boundary_length = len(boundary_cells)
        elif boundary_length != len(boundary_cells):
            raise BsdlError(
                f"BOUNDARY_LENGTH {boundary_length} != number of "
                f"BOUNDARY_REGISTER cells {len(boundary_cells)}")

    return {
        "name": name,
        "ir_length": ir_length,
        "ir_capture": ir_capture,
        "opcodes": opcodes,
        "idcode_value": hex(idcode_value) if idcode_value is not None else None,
        "idcode_mask": hex(idcode_mask) if idcode_mask is not None else None,
        "boundary_length": boundary_length,
        "boundary_cells": boundary_cells,
    }
