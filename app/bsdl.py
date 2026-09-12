"""Minimal BSDL (IEEE 1149.1) parser.

Extracts only what chain reconstruction needs:
  - entity name
  - INSTRUCTION_LENGTH
  - INSTRUCTION_OPCODE  (IDCODE / BYPASS / SAMPLE / PRELOAD / ...)
  - INSTRUCTION_CAPTURE (may contain X don't-care bits)
  - IDCODE_REGISTER     (32-bit value, X bits become mask 0)
  - BOUNDARY_LENGTH

Bit-string convention used across the whole project:
  * BSDL writes patterns MSB-left ("0001" means bit3..bit0 = 0,0,0,1).
  * The inference engine works in *shift-out order*: index 0 is the first
    bit seen on TDO, which for a register shifted LSB-first is bit 0.
  * `parse_bsdl` returns patterns in BSDL order; conversion to shift order
    happens in `app.models.device_from_in` (a plain reverse).
"""
from __future__ import annotations

import re


class BsdlError(ValueError):
    pass


def _attr(text: str, name: str) -> str | None:
    m = re.search(
        r"attribute\s+" + re.escape(name) + r"\s+of\s+\w+\s*:\s*entity\s+is\s+(.*?);",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    return m.group(1) if m else None


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

    return {
        "name": name,
        "ir_length": ir_length,
        "ir_capture": ir_capture,
        "opcodes": opcodes,
        "idcode_value": hex(idcode_value) if idcode_value is not None else None,
        "idcode_mask": hex(idcode_mask) if idcode_mask is not None else None,
        "boundary_length": boundary_length,
    }
