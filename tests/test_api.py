"""End-to-end tests: simulate an ideal JTAG chain, capture bit streams,
and verify the API reconstructs the chain."""
import os

import pytest

os.environ["JTAG_RECON_DB"] = ":memory:"  # overridden per-test below via tmp file


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("JTAG_RECON_DB", str(tmp_path / "test.db"))
    import importlib
    import app.db as db
    importlib.reload(db)
    db.init()
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    return TestClient(main.app)


# ------------------------------------------------------------------ BSDL under test

BSDL_A = """
entity DEVA is
attribute INSTRUCTION_LENGTH of DEVA : entity is 4;
attribute INSTRUCTION_OPCODE of DEVA : entity is
  "IDCODE (0010), BYPASS (1111), SAMPLE (0001), PRELOAD (0001), EXTEST (0000)";
attribute INSTRUCTION_CAPTURE of DEVA : entity is "0001";
attribute IDCODE_REGISTER of DEVA : entity is
  "0000" & "0011011001000001" & "00001001001" & "1";
attribute BOUNDARY_LENGTH of DEVA : entity is 8;
end DEVA;
"""

BSDL_B = """
entity DEVB is
attribute INSTRUCTION_LENGTH of DEVB : entity is 6;
attribute INSTRUCTION_OPCODE of DEVB : entity is
  "BYPASS (111111), SAMPLE (000011), PRELOAD (000011), EXTEST (000000)";
attribute INSTRUCTION_CAPTURE of DEVB : entity is "000001";
attribute BOUNDARY_LENGTH of DEVB : entity is 12;
end DEVB;
"""

IDCODE_A = 0x03641093  # matches BSDL_A fields


# ------------------------------------------------------------------ chain simulator

def shift(capture_bits: str, tdi: str) -> str:
    """Ideal scan: TDO = captured bits, then TDI delayed by the register length."""
    L = len(capture_bits)
    if len(tdi) <= L:
        return capture_bits[: len(tdi)]
    return capture_bits + tdi[: len(tdi) - L]


def ir_capture_region(chain_tdi_order):
    """IR capture bits in TDO order (TDO-side device first, LSB first)."""
    return "".join(d["capture_shift"] for d in reversed(chain_tdi_order))


DEVA = {"capture_shift": "1000", "ir_len": 4}
DEVB = {"capture_shift": "100000", "ir_len": 6}
DEVC_UNKNOWN = {"capture_shift": "10001", "ir_len": 5}  # no BSDL available


def make_captures(chain_tdi_order, idcode_devices=()):
    """chain_tdi_order: list of device dicts, TDI-side first.
    idcode_devices: indices (chain order) that have an IDCODE register."""
    tdi_ir = "1011001010110010101100101011001010110010"
    tdo_ir = shift(ir_capture_region(chain_tdi_order), tdi_ir)

    tdi_bp = "111000101100"
    tdo_bp = shift("0" * len(chain_tdi_order), tdi_bp)

    # build IDCODE DR capture: TDO-side first
    idc_bits = ""
    for idx in range(len(chain_tdi_order) - 1, -1, -1):
        if idx in idcode_devices:
            idc_bits += format(IDCODE_A, "032b")[::-1]  # LSB first
        else:
            idc_bits += "0"
    tdi_id = "011011001010110010110010101100101100"
    tdo_id = shift(idc_bits, tdi_id)

    return [
        {"kind": "ir", "instruction": None, "tdi": tdi_ir, "tdo": tdo_ir},
        {"kind": "dr", "instruction": "BYPASS", "tdi": tdi_bp, "tdo": tdo_bp},
        {"kind": "dr", "instruction": "IDCODE", "tdi": tdi_id, "tdo": tdo_id},
    ]


def upload_devices(client):
    ida = client.post("/devices/bsdl", json={"bsdl_text": BSDL_A}).json()["device_id"]
    idb = client.post("/devices/bsdl", json={"bsdl_text": BSDL_B}).json()["device_id"]
    return ida, idb


# ------------------------------------------------------------------ tests

def test_bsdl_parsing(client):
    r = client.post("/devices/bsdl", json={"bsdl_text": BSDL_A})
    assert r.status_code == 201
    m = r.json()["model"]
    assert m["name"] == "DEVA"
    assert m["ir_length"] == 4
    assert m["ir_capture"] == "0001"
    assert m["opcodes"]["IDCODE"] == "0010"
    assert int(m["idcode_value"], 16) == IDCODE_A
    assert m["boundary_length"] == 8


def test_infer_two_device_chain(client):
    ida, idb = upload_devices(client)
    captures = make_captures([DEVA, DEVB], idcode_devices=(0,))
    r = client.post("/infer", json={
        "session": "t1",
        "devices": [{"device_id": ida}, {"device_id": idb}],
        "captures": captures,
    })
    assert r.status_code == 201, r.text
    body = r.json()
    top = body["candidates"][0]
    assert [s["device"] for s in top["chain"]] == ["DEVA", "DEVB"]
    assert [s["ir_length"] for s in top["chain"]] == [4, 6]
    assert any("IDCODE match: DEVA" in c for c in top["constraints"])
    assert any("device count 2 confirmed" in c for c in top["constraints"])
    assert top["chain"][0]["idcode_observed"] == hex(IDCODE_A)
    assert not body["checks"]["tdo_constant"]["flag"]
    assert not body["checks"]["overall_offset_by_one"]["flag"]
    assert not body["checks"]["indistinguishable_candidates"]["flag"]


def test_missing_device_detected(client):
    ida, idb = upload_devices(client)
    # real chain is DEVA -> unknown(5) -> DEVB, but only A and B are known
    captures = make_captures([DEVA, DEVC_UNKNOWN, DEVB], idcode_devices=(0,))
    r = client.post("/infer", json={
        "devices": [{"device_id": ida}, {"device_id": idb}],
        "captures": captures,
        "max_unknown": 1,
    })
    top = r.json()["candidates"][0]
    assert [s["status"] for s in top["chain"]] == ["identified", "unknown", "identified"]
    assert top["chain"][1]["ir_length"] == 5
    assert any("unidentified device" in u for u in top["unexplained"])


def test_locks_and_unstable_bits(client):
    ida, idb = upload_devices(client)
    captures = make_captures([DEVA, DEVB], idcode_devices=(0,))
    # corrupt one IR capture bit and mark it unreliable
    bad = list(captures[0]["tdo"])
    bad[3] = "1" if bad[3] == "0" else "0"
    captures[0]["tdo"] = "".join(bad)
    r = client.post("/infer", json={
        "devices": [{"device_id": ida}, {"device_id": idb}],
        "captures": captures,
        "unreliable": [{"capture_index": 0, "start": 3, "end": 4}],
        "locks": [{"position": 0, "device": "DEVA"}, {"position": 1, "device": "DEVB"}],
    })
    top = r.json()["candidates"][0]
    assert [s["device"] for s in top["chain"]] == ["DEVA", "DEVB"]


def test_tdo_constant_flagged(client):
    ida, idb = upload_devices(client)
    r = client.post("/infer", json={
        "devices": [{"device_id": ida}, {"device_id": idb}],
        "captures": [{"kind": "ir", "instruction": None,
                      "tdi": "10101010", "tdo": "11111111"}],
    })
    assert r.json()["checks"]["tdo_constant"]["flag"]


def test_idcode_mask_mismatch_flagged(client):
    ida, idb = upload_devices(client)
    captures = make_captures([DEVA, DEVB], idcode_devices=(0,))
    # flip one IDCODE bit (TDO position 1 = DEVB bypass bit, position 1..32 = DEVA idcode)
    tdo = list(captures[2]["tdo"])
    tdo[5] = "1" if tdo[5] == "0" else "0"
    captures[2]["tdo"] = "".join(tdo)
    r = client.post("/infer", json={
        "devices": [{"device_id": ida}, {"device_id": idb}],
        "captures": captures,
    })
    body = r.json()
    assert body["checks"]["idcode_mask_mismatch"]["flag"]
    assert any("IDCODE mismatch" in u for u in body["candidates"][0]["unexplained"])


def test_svf_and_json_export(client):
    ida, idb = upload_devices(client)
    captures = make_captures([DEVA, DEVB], idcode_devices=(0,))
    r = client.post("/infer", json={
        "devices": [{"device_id": ida}, {"device_id": idb}],
        "captures": captures,
    })
    vid = r.json()["version_id"]

    r = client.get(f"/versions/{vid}/svf")
    assert r.status_code == 200
    text = r.text
    # EXTEST may only appear inside the safety comment, never in a command
    assert all("EXTEST" not in line for line in text.splitlines()
               if not line.startswith("!"))
    assert "SIR 10 TDI (3FF) TDO (041) MASK (3FF);" in text          # all BYPASS
    assert "SDR 2 TDI (0) TDO (0) MASK (3);" in text                # bypass count
    assert "SDR 33 TDI (000000000) TDO (006C82126) MASK (1FFFFFFFF);" in text  # idcode
    assert "SDR 20 TDI (00000) TDO (00000) MASK (00000);" in text   # SAMPLE/PRELOAD

    r = client.get(f"/versions/{vid}/export")
    body = r.json()
    assert body["format"] == "jtag-chain-definition"
    assert [s["device"] for s in body["chain"]] == ["DEVA", "DEVB"]

    # stored raw captures and versions are retrievable
    assert len(client.get("/captures").json()) == 3
    assert client.get(f"/versions/{vid}").status_code == 200
