"""Pydantic request/response schemas."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .inference import Device


class BsdlUpload(BaseModel):
    bsdl_text: str
    name: str | None = None  # optional friendly name override


class DeviceModelIn(BaseModel):
    """Device description using BSDL bit order (patterns MSB-left)."""

    name: str
    ir_length: int = Field(gt=1, le=64)
    ir_capture: str = ""  # MSB-left, may contain X; empty = unknown
    opcodes: dict[str, str] = {}
    idcode_value: str | None = None  # hex, e.g. "0x03641093"
    idcode_mask: str | None = None
    boundary_length: int | None = None

    @field_validator("ir_capture")
    @classmethod
    def _cap_bits(cls, v: str) -> str:
        if v and not re.fullmatch(r"[01xX]+", v):
            raise ValueError("ir_capture must contain only 0/1/X")
        return v.upper()


class CaptureIn(BaseModel):
    """One raw scan. Bit strings are in wire order: index 0 = first bit
    clocked on that wire (for TDO: first bit out of the chain)."""

    kind: Literal["ir", "dr"]
    instruction: str | None = None  # for dr scans: BYPASS / IDCODE / SAMPLE / PRELOAD
    tdi: str
    tdo: str

    @field_validator("tdi", "tdo")
    @classmethod
    def _bits(cls, v: str) -> str:
        if not re.fullmatch(r"[01]*", v):
            raise ValueError("bit streams must contain only 0/1")
        return v


class DeviceRef(BaseModel):
    device_id: int | None = None  # id of a previously uploaded BSDL
    model: DeviceModelIn | None = None  # or an inline model
    count: int = Field(default=1, ge=1, le=8)  # how many of this device may appear


class LockIn(BaseModel):
    position: int = Field(ge=0)  # chain position, 0 = closest to TDI
    device: str  # device name, or "unknown" to pin an unidentified slot


class UnreliableIn(BaseModel):
    capture_index: int = Field(ge=0)
    start: int = Field(ge=0)  # TDO bit range [start, end) treated as don't-care
    end: int = Field(ge=0)


class InferRequest(BaseModel):
    session: str = "default"
    bit_order: Literal["lsb_first", "msb_first"] = "lsb_first"
    devices: list[DeviceRef] = []
    captures: list[CaptureIn]
    locks: list[LockIn] = []
    unreliable: list[UnreliableIn] = []
    max_unknown: int = Field(default=2, ge=0, le=4)
    note: str = ""


def device_from_in(m: DeviceModelIn) -> Device:
    """Convert the API model (BSDL bit order) to an inference Device
    (shift-out order)."""
    capture = m.ir_capture[::-1] if m.ir_capture else "X" * m.ir_length
    value = int(m.idcode_value, 16) if m.idcode_value else None
    mask = int(m.idcode_mask, 16) if m.idcode_mask else None
    if value is not None and mask is None:
        mask = 0xFFFFFFFF
    return Device(
        name=m.name,
        ir_length=m.ir_length,
        ir_capture=capture,
        opcodes={k.upper(): v for k, v in m.opcodes.items()},
        idcode_value=value,
        idcode_mask=mask,
        boundary_length=m.boundary_length,
    )
