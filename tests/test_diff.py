"""End-to-end tests for chain-version diff and bit-map migration."""
import os

import pytest

os.environ["JTAG_RECON_DB"] = ":memory:"

IDCODE_A = 0x03641093

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

BSDL_C = """
entity DEVC is
attribute INSTRUCTION_LENGTH of DEVC : entity is 5;
attribute INSTRUCTION_OPCODE of DEVC : entity is
  "BYPASS (11111), SAMPLE (00001), PRELOAD (00001)";
attribute INSTRUCTION_CAPTURE of DEVC : entity is "00001";
attribute BOUNDARY_LENGTH of DEVC : entity is 6;
end DEVC;
"""


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


# ------------------------------------------------------------------ simulator

def shift(capture_bits, tdi):
    L = len(capture_bits)
    return capture_bits + tdi[: max(0, len(tdi) - L)]


DEVS = {
    "DEVA": {"capture_shift": "1000", "ir_len": 4, "bl": 8, "idcode": True},
    "DEVB": {"capture_shift": "100000", "ir_len": 6, "bl": 12, "idcode": False},
    "DEVC": {"capture_shift": "10000", "ir_len": 5, "bl": 6, "idcode": False},
}

TDI_IR = "1011001010110010101100101011001010110010"
TDI_BP = "111000101100"
TDI_ID = "011011001010110010110010101100101100"
IDC_SHIFT = format(IDCODE_A, "032b")[::-1]


def make_captures(names_tdi_order):
    ir = "".join(DEVS[n]["capture_shift"] for n in reversed(names_tdi_order))
    idc = ""
    for n in reversed(names_tdi_order):
        idc += IDC_SHIFT if DEVS[n]["idcode"] else "0"
    return [
        {"kind": "ir", "instruction": None,
         "tdi": TDI_IR, "tdo": shift(ir, TDI_IR)},
        {"kind": "dr", "instruction": "BYPASS",
         "tdi": TDI_BP, "tdo": shift("0" * len(names_tdi_order), TDI_BP)},
        {"kind": "dr", "instruction": "IDCODE",
         "tdi": TDI_ID, "tdo": shift(idc, TDI_ID)},
    ]


def upload(client):
    ids = {}
    for name, text in (("DEVA", BSDL_A), ("DEVB", BSDL_B), ("DEVC", BSDL_C)):
        ids[name] = client.post("/devices/bsdl",
                                json={"bsdl_text": text}).json()["device_id"]
    return ids


def infer(client, session, names, ids, locks=None, **extra):
    r = client.post("/infer", json={
        "session": session,
        "devices": [{"device_id": ids[n]} for n in names],
        "captures": make_captures(names),
        **({"locks": locks} if locks else {}),
        **extra,
    })
    assert r.status_code == 201, r.text
    return r.json()["version_id"]


def regmap(body, register):
    for m in body["register_mappings"]:
        if m["register"] == register:
            return m
    raise AssertionError(register)


# ------------------------------------------------------------------ tests

def test_identical_versions_no_events(client):
    ids = upload(client)
    v1 = infer(client, "s1", ["DEVA", "DEVB"], ids)
    v2 = infer(client, "s1", ["DEVA", "DEVB"], ids)
    r = client.post("/diff", json={
        "session": "s1", "source_version_id": v1,
        "target_version_id": v2})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["alignment"]["status"] == "unique"
    assert body["alignment"]["events"] == []
    assert [p["basis"] for p in body["alignment"]["pairs"]]
    for m in body["register_mappings"]:
        assert m["counts"]["remapped"] == 0
        assert m["counts"]["dropped"] == 0
        assert m["counts"]["added"] == 0
    # all three historical captures of the source stay interpretable
    assert body["summary"]["samples_total"] == 3
    assert body["summary"]["samples_contradicting"] == 0
    assert body["summary"]["samples_not_interpretable"] == 0
    for it in body["sample_migration"]["items"]:
        assert it["verdict"] in ("interpretable", "interpretable_with_gaps")
    # batch persisted independently, versions/captures untouched
    bid = body["batch_id"]
    assert client.get(f"/diff/{bid}").json()["result"]["summary"][
        "samples_total"] == 3
    assert any(b["id"] == bid for b in client.get("/diff?session=s1").json())
    assert len(client.get("/captures").json()) == 6  # 3 per inference


def test_device_insertion_bitmap_and_sample_migration(client):
    ids = upload(client)
    v_old = infer(client, "s2", ["DEVA", "DEVB"], ids)
    v_new = infer(client, "s2", ["DEVA", "DEVC", "DEVB"], ids)
    r = client.post("/diff", json={
        "session": "s2", "source_version_id": v_old,
        "target_version_id": v_new})
    assert r.status_code == 201, r.text
    body = r.json()

    al = body["alignment"]
    assert al["status"] == "partial"
    assert [d["device"] for d in al["inserted"]] == ["DEVC"]
    assert al["deleted"] == []
    assert not al["reordered"]
    assert any(e["type"] == "device_inserted" and e["device"] == "DEVC"
               for e in al["events"])
    pairs = {(p["source_device"], p["target_device"]): p
             for p in al["pairs"]}
    assert pairs[("DEVA", "DEVA")]["source_position"] == 0
    assert pairs[("DEVA", "DEVA")]["target_position"] == 0
    assert pairs[("DEVB", "DEVB")]["source_position"] == 1
    assert pairs[("DEVB", "DEVB")]["target_position"] == 2

    # IR: 10 -> 15 bits. Wire order (TDO-side first): DEVB occupies bits
    # 0..5 in both versions; DEVA shifts from 6..9 to 11..14; the inserted
    # DEVC occupies target bits 6..10.
    ir = regmap(body, "IR")
    assert (ir["source_length"], ir["target_length"]) == (10, 15)
    assert ir["counts"]["added"] == 5
    devb = [r for r in ir["bits"] if r["device"] == "DEVB"
            and r["source_bit"] is not None and r["target_bit"] is not None]
    assert all(r["target_bit"] == r["source_bit"] for r in devb)
    deva = [r for r in ir["bits"] if r["device"] == "DEVA"]
    assert all(r["target_bit"] == r["source_bit"] + 5
               and r["status"] == "remapped" for r in deva)
    devc = [r for r in ir["changed_bits"] if r["device"] == "DEVC"]
    assert len(devc) == 5 and all(r["target_bit"] in range(6, 11) for r in devc)

    # BYPASS: 2 -> 3 bits; old samples lack the new device bit -> gap
    bp = regmap(body, "DR:BYPASS")
    assert (bp["source_length"], bp["target_length"]) == (2, 3)

    # IDCODE: DEVA contributes 32 bits, DEVB and the inserted DEVC one
    # bypass bit each: 33 -> 34 bits.
    idc = regmap(body, "DR:IDCODE")
    assert idc["target_length"] == 34

    # the OLD captured bit streams replayed on the NEW target chain:
    # IR (10 vs 15) differs by 5 bits -> length mismatch, not interpretable;
    # BYPASS (2 vs 3) / IDCODE (32 vs 33) differ by one bit -> the single
    # new bit was never observed, interpretable with a gap.
    verdicts = {(it["kind"], it["instruction"]): it
                for it in body["sample_migration"]["items"]}
    ir_item = verdicts[("ir", None)]
    assert ir_item["verdict"] == "not_interpretable"
    assert "register_length_mismatch" in ir_item["reasons"]
    bp_item = verdicts[("dr", "BYPASS")]
    assert bp_item["verdict"] == "interpretable_with_gaps"
    assert bp_item["missing_bits"]["count"] == 1
    assert any(e["target_position"] == 1 and e["device"] == "DEVC"
               for e in bp_item["added_bits"])
    id_item = verdicts[("dr", "IDCODE")]
    assert id_item["verdict"] == "interpretable_with_gaps"
    assert id_item["missing_bits"]["count"] == 1
    assert body["summary"]["samples_not_interpretable"] == 1
    assert body["summary"]["samples_interpretable"] == 2


def test_device_deletion_missing_target_bits(client):
    ids = upload(client)
    v_old = infer(client, "s3", ["DEVA", "DEVC", "DEVB"], ids)
    v_new = infer(client, "s3", ["DEVA", "DEVB"], ids)
    r = client.post("/diff", json={
        "session": "s3", "source_version_id": v_old,
        "target_version_id": v_new})
    body = r.json()
    assert [d["device"] for d in body["alignment"]["deleted"]] == ["DEVC"]
    ir = regmap(body, "IR")
    assert ir["counts"]["dropped"] == 5
    # TDO order is DEVB(0..5), DEVC(6..10), DEVA(11..14) in the old chain;
    # after deleting DEVC: DEVB(0..5), DEVA(6..9) -> DEVA collapses 11..14
    deva = [r for r in ir["changed_bits"] if r["device"] == "DEVA"]
    assert {r["source_bit"]: r["target_bit"] for r in deva} == {
        11: 6, 12: 7, 13: 8, 14: 9}


def test_reorder_detected(client):
    ids = upload(client)
    v_old = infer(client, "s4", ["DEVA", "DEVB"], ids)
    # physically reversed reinstallation: target captures come from [DEVB, DEVA]
    r = client.post("/infer", json={
        "session": "s4",
        "devices": [{"device_id": ids["DEVA"]}, {"device_id": ids["DEVB"]}],
        "captures": make_captures(["DEVB", "DEVA"])})
    assert r.status_code == 201, r.text
    top = r.json()["candidates"][0]["chain"]
    assert [s["device"] for s in top] == ["DEVB", "DEVA"]
    v_new = r.json()["version_id"]
    r = client.post("/diff", json={
        "session": "s4", "source_version_id": v_old,
        "target_version_id": v_new})
    body = r.json()
    assert body["alignment"]["reordered"]
    assert any(e["type"] == "devices_reordered"
               for e in body["alignment"]["events"])
    # the old captured IR content contradicts the swapped target chain:
    # echo alignment still holds but the captured IR bits do not fit the
    # target IR layout -> cannot be interpreted, with per-bit evidence
    ir_item = next(it for it in body["sample_migration"]["items"]
                   if it["kind"] == "ir")
    assert ir_item["verdict"] == "contradicts_target"
    assert "ir_capture_mismatch" in ir_item["reasons"]
    assert "content_contradicts_target" in ir_item["reasons"]
    assert ir_item["mismatch_bits"]
    assert {"DEVA", "DEVB"} & set(ir_item["affected_devices"])


def test_ambiguous_alignment_keeps_alternatives(client):
    """Two identical unidentified slots in the source can pair with two
    same-length known devices in the target -> alignment not unique."""
    ids = upload(client)
    # source: known DEVB + two unidentified IR-4 devices ("1000" capture)
    names = ["DEVA", "DEVA", "DEVB"]
    r = client.post("/infer", json={
        "session": "s5",
        "devices": [{"device_id": ids["DEVB"]}],
        "captures": make_captures(names), "max_unknown": 2})
    assert r.status_code == 201, r.text
    top = r.json()["candidates"][0]["chain"]
    assert sum(1 for s in top if s["status"] == "unknown") == 2
    v_old = r.json()["version_id"]
    # target: both IR-4 devices identified as DEVA (identical signatures)
    r = client.post("/infer", json={
        "session": "s5",
        "devices": [{"device_id": ids["DEVA"], "count": 2},
                    {"device_id": ids["DEVB"]}],
        "captures": make_captures(names)})
    assert r.status_code == 201, r.text
    v_new = r.json()["version_id"]

    r = client.post("/diff", json={
        "session": "s5", "source_version_id": v_old,
        "target_version_id": v_new})
    body = r.json()
    assert body["alignment"]["status"] == "ambiguous"
    assert len(body["alignment"]["candidate_alignments"]) >= 2
    amb = [p for p in body["alignment"]["pairs"] if p["alternatives"]]
    assert amb
    assert any(any("ir_length" in a["basis"] or "capture_signature" in a["basis"]
                   for a in p["alternatives"]) for p in amb)


def test_cross_session_and_bad_refs_rejected(client):
    ids = upload(client)
    v1 = infer(client, "sA", ["DEVA", "DEVB"], ids)
    v2 = infer(client, "sB", ["DEVA", "DEVB"], ids)
    r = client.post("/diff", json={
        "session": "sA", "source_version_id": v1,
        "target_version_id": v2})
    assert r.status_code == 422
    r = client.post("/diff", json={
        "session": "sA", "source_version_id": v1,
        "target_version_id": 999})
    assert r.status_code == 404
    r = client.post("/diff", json={
        "session": "sA", "source_version_id": v1,
        "target_version_id": v1})
    assert r.status_code == 422  # same version and candidate


def test_consistency_batch_samples_migrated(client):
    ids = upload(client)
    v_old = infer(client, "s6", ["DEVA", "DEVB"], ids)
    # a saved consistency batch on the old version
    ir_old = "".join(DEVS[n]["capture_shift"]
                     for n in reversed(["DEVA", "DEVB"]))
    run = {"label": "cold-1", "kind": "ir", "instruction": None,
           "tdi": TDI_IR, "tdo": shift(ir_old, TDI_IR)}
    r = client.post("/consistency", json={
        "session": "s6", "version_id": v_old, "runs": [run]})
    assert r.status_code == 201, r.text
    bid = r.json()["batch_id"]

    v_new = infer(client, "s6", ["DEVA", "DEVC", "DEVB"], ids)
    r = client.post("/diff", json={
        "session": "s6", "source_version_id": v_old,
        "target_version_id": v_new,
        "include_version_captures": False,
        "consistency_batch_ids": [bid]})
    body = r.json()
    assert body["sample_migration"]["selection"] == {
        "version_captures": [], "consistency_batches": [bid],
        "capture_ids": []}
    assert body["summary"]["samples_total"] == 1
    item = body["sample_migration"]["items"][0]
    assert item["origin"]["consistency_batch_id"] == bid
    assert item["origin"]["label"] == "cold-1"
    summ = body["sample_migration"]["batch_summary"][0]
    assert summ["consistency_batch_id"] == bid
    assert summ["not_interpretable"] == 1


def test_boundary_length_change_and_custom_instruction(client):
    ids = upload(client)
    # source: DEVA+DEVB (boundary 8+12=20); target inserts DEVC (6) -> 26
    v_old = infer(client, "s7", ["DEVA", "DEVB"], ids)
    v_new = infer(client, "s7", ["DEVA", "DEVC", "DEVB"], ids)
    r = client.post("/diff", json={
        "session": "s7", "source_version_id": v_old,
        "target_version_id": v_new})
    body = r.json()
    smp = regmap(body, "DR:SAMPLE")
    assert (smp["source_length"], smp["target_length"]) == (20, 26)
    assert smp["counts"]["added"] == 6
    assert any(e["type"] == "boundary_length_changed"
               for e in body["alignment"]["events"]) is False  # lengths per device unchanged
    assert any(e["type"] == "device_inserted"
               for e in body["alignment"]["events"])


def test_same_version_two_candidates_allowed(client):
    ids = upload(client)
    v = infer(client, "s8", ["DEVA", "DEVB"], ids)
    # identical candidate pair is rejected; different candidate index works
    # only when the version actually has >1 candidates.
    ver = client.get(f"/versions/{v}").json()
    nc = len(ver["result"]["candidates"])
    if nc > 1:
        r = client.post("/diff", json={
            "session": "s8", "source_version_id": v, "source_candidate": 0,
            "target_version_id": v, "target_candidate": 1})
        assert r.status_code == 201, r.text


def test_diff_batch_wrong_version_rejected(client):
    ids = upload(client)
    v1 = infer(client, "s9", ["DEVA", "DEVB"], ids)
    v2 = infer(client, "s9", ["DEVA", "DEVC", "DEVB"], ids)
    run = {"label": "r", "kind": "ir", "instruction": None,
           "tdi": TDI_IR,
           "tdo": shift("".join(DEVS[n]["capture_shift"]
                                for n in reversed(["DEVA", "DEVB"])), TDI_IR)}
    bid = client.post("/consistency", json={
        "session": "s9", "version_id": v1, "runs": [run]}).json()["batch_id"]
    r = client.post("/diff", json={
        "session": "s9", "source_version_id": v2,
        "target_version_id": v1,
        "consistency_batch_ids": [bid]})
    assert r.status_code == 422
    assert "version" in r.json()["detail"]


def test_unknown_candidate_and_cross_session_capture(client):
    ids = upload(client)
    v1 = infer(client, "sA1", ["DEVA", "DEVB"], ids)
    v2 = infer(client, "sA1", ["DEVA", "DEVC", "DEVB"], ids)
    r = client.post("/diff", json={
        "session": "sA1", "source_version_id": v1, "source_candidate": 999,
        "target_version_id": v2})
    assert r.status_code == 404
    # a capture id from another session is rejected
    infer(client, "sA2", ["DEVA", "DEVB"], ids)
    cid = client.get("/captures?session=sA2").json()[0]["id"]
    r = client.post("/diff", json={
        "session": "sA1", "source_version_id": v1,
        "target_version_id": v2, "capture_ids": [cid]})
    assert r.status_code == 422
