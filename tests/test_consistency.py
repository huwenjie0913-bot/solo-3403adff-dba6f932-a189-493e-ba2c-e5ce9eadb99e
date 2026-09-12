"""End-to-end tests for the repeat-sample consistency / intermittent-fault
localization module."""
import os

import pytest

os.environ["JTAG_RECON_DB"] = ":memory:"

IR_CAP_SHIFT = "1000001000"   # TDO order: DEVB(6, position 1) then DEVA(4, 0)
IDCODE_A = 0x03641093
IDC_SHIFT = format(IDCODE_A, "032b")[::-1]  # LSB first
TDI_IR = "1011001010110010101100101011001010110010"


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


def _version(client, session="cons"):
    BSDL_A = """
entity DEVA is
attribute INSTRUCTION_LENGTH of DEVA : entity is 4;
attribute INSTRUCTION_OPCODE of DEVA : entity is
  "IDCODE (0010), BYPASS (1111), SAMPLE (0001), PRELOAD (0001)";
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
  "BYPASS (111111), SAMPLE (000011), PRELOAD (000011)";
attribute INSTRUCTION_CAPTURE of DEVB : entity is "000001";
attribute BOUNDARY_LENGTH of DEVB : entity is 12;
end DEVB;
"""
    ida = client.post("/devices/bsdl", json={"bsdl_text": BSDL_A}).json()["device_id"]
    idb = client.post("/devices/bsdl", json={"bsdl_text": BSDL_B}).json()["device_id"]

    def shift(capture_bits, tdi):
        L = len(capture_bits)
        return capture_bits + tdi[:len(tdi) - L]

    tdi_bp = "111000101100"
    tdi_id = "011011001010110010110010101100101100"
    captures = [
        {"kind": "ir", "instruction": None, "tdi": TDI_IR,
         "tdo": shift(IR_CAP_SHIFT, TDI_IR)},
        {"kind": "dr", "instruction": "BYPASS", "tdi": tdi_bp,
         "tdo": shift("00", tdi_bp)},
        {"kind": "dr", "instruction": "IDCODE", "tdi": tdi_id,
         "tdo": shift("0" + IDC_SHIFT, tdi_id)},  # DEVB bypass then DEVA
    ]
    r = client.post("/infer", json={
        "session": session,
        "devices": [{"device_id": ida}, {"device_id": idb}],
        "captures": captures,
    })
    assert r.status_code == 201, r.text
    return r.json()["version_id"]


def ir_run(label, tdo_capture, tdi=TDI_IR):
    """Build an IR run whose capture region (10 bits) is tdo_capture."""
    return {"label": label, "kind": "ir", "instruction": None,
            "tdi": tdi, "tdo": tdo_capture + tdi[:len(tdi) - 10]}


def _group(body, kind, instruction=None):
    for g in body["groups"]:
        if g["kind"] == kind and g["instruction"] == instruction:
            return g
    raise AssertionError(f"group {kind}/{instruction} not found")


def _diag_kinds(group):
    return [d["kind"] for d in group["diagnoses"]]


# ------------------------------------------------------------------ tests

def test_stable_runs_no_diagnosis(client):
    vid = _version(client)
    runs = [ir_run(f"r{i}", IR_CAP_SHIFT) for i in range(3)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    assert r.status_code == 201, r.text
    g = _group(r.json(), "ir")
    assert g["mapping"] == "mapped"
    assert g["register_length"] == 10
    assert all(run["status"] == "aligned" for run in g["runs"])
    assert g["summary"]["unstable_bits"] == 0
    assert g["diagnoses"] == []
    # segments: TDO-side first -> DEVB (chain position 1), then DEVA (0)
    assert [s["device"] for s in g["segments"]] == ["DEVB", "DEVA"]
    assert [(s["start_bit"], s["length"]) for s in g["segments"]] == [(0, 6), (6, 4)]
    # every bit lists its stable value and 3 observations
    for b in g["bits"]:
        assert b["stable"] == IR_CAP_SHIFT[b["chain_bit"]]
        assert b["count_0"] + b["count_1"] == 3
        assert b["toggles"] == 0


def test_intermittent_single_device_flip(client):
    vid = _version(client)
    bad = list(IR_CAP_SHIFT)
    bad[7] = "1" if bad[7] == "0" else "0"  # chain bit 7 = DEVA local bit 1
    runs = [ir_run("good1", IR_CAP_SHIFT),
            ir_run("flaky", "".join(bad)),
            ir_run("good2", IR_CAP_SHIFT),
            ir_run("flaky2", "".join(bad)),
            ir_run("good3", IR_CAP_SHIFT)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    assert r.status_code == 201, r.text
    g = _group(r.json(), "ir")
    assert "intermittent_device_toggle" in _diag_kinds(g)
    diag = next(d for d in g["diagnoses"]
                if d["kind"] == "intermittent_device_toggle")
    # TDO bits 0..5 = DEVB (position 1), bits 6..9 = DEVA (position 0)
    assert diag["affected_devices"] == ["DEVA"]
    assert diag["affected_positions"] == [0]
    assert diag["register_bits"] == [7]
    assert diag["runs"] == ["flaky", "flaky2"]
    assert diag["first_anomaly"]["run"] == "flaky"
    assert diag["first_anomaly"]["chain_bit"] == 7
    # raw supporting indices are present
    ev = diag["evidence"]
    assert ev["bits"][0]["register"] == "IR"
    assert ev["samples"][0]["tdo_indices"] == [7]
    # the bit statistics itself
    bit = g["bits"][7]
    assert bit["stable"] == IR_CAP_SHIFT[7]
    assert bit["count_0"] + bit["count_1"] == 5
    assert bit["toggles"] >= 2


def test_isolated_single_sample_anomaly(client):
    vid = _version(client)
    bad = list(IR_CAP_SHIFT)
    bad[8] = "1" if bad[8] == "0" else "0"  # DEVA local bit 2, one run only
    runs = [ir_run("g1", IR_CAP_SHIFT),
            ir_run("g2", IR_CAP_SHIFT),
            ir_run("glitch", "".join(bad)),
            ir_run("g3", IR_CAP_SHIFT)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    g = _group(r.json(), "ir")
    assert _diag_kinds(g) == ["isolated_sample_anomaly"]
    diag = g["diagnoses"][0]
    assert diag["runs"] == ["glitch"]
    assert diag["affected_devices"] == ["DEVA"]
    assert diag["register_bits"] == [8]


def test_tdo_constant_run_excluded(client):
    vid = _version(client)
    # capture region and echo all zero: TDO never toggles in that run
    runs = [ir_run("ok1", IR_CAP_SHIFT),
            {"label": "dead", "kind": "ir", "instruction": None,
             "tdi": TDI_IR, "tdo": "0" * len(TDI_IR)},
            ir_run("ok2", IR_CAP_SHIFT)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    assert r.status_code == 201, r.text
    g = _group(r.json(), "ir")
    dead = next(x for x in g["runs"] if x["label"] == "dead")
    assert dead["status"] == "excluded"
    assert dead["excluded_reason"] == "tdo_constant"
    diag = next(d for d in g["diagnoses"] if d["kind"] == "tdo_constant")
    assert diag["runs"] == ["dead"]
    assert set(diag["affected_devices"]) == {"DEVA", "DEVB"}
    # excluded run does not participate in statistics
    for b in g["bits"]:
        assert "dead" not in b["observed_runs"]
    assert g["summary"]["aligned"] == 2
    assert g["summary"]["excluded"] == 1
    assert g["summary"]["unstable_bits"] == 0


def test_segment_fixed_offset(client):
    vid = _version(client)
    # capture content shifted one bit toward TDO, echo still at TDO bit 10
    shifted = TDI_IR[0] + IR_CAP_SHIFT[:-1] + TDI_IR[:len(TDI_IR) - 10]
    runs = [ir_run("g1", IR_CAP_SHIFT),
            ir_run("g2", IR_CAP_SHIFT),
            ir_run("g3", IR_CAP_SHIFT),
            ir_run("off", shifted)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    g = _group(r.json(), "ir")
    diag = next((d for d in g["diagnoses"]
                 if d["kind"] == "segment_fixed_offset"), None)
    assert diag is not None, g["diagnoses"]
    assert diag["runs"] == ["off"]
    assert diag["shift_bits"] == -1
    assert "DEVA" in diag["affected_devices"]
    assert diag["register_bits"]  # non-empty evidence


def test_register_length_mismatch_run_excluded(client):
    vid = _version(client)
    # insert 5 extra fixed bits before the capture region -> echo at 15
    padded = "00000" + IR_CAP_SHIFT + TDI_IR[:len(TDI_IR) - 15]
    runs = [ir_run("g1", IR_CAP_SHIFT),
            ir_run("g2", IR_CAP_SHIFT),
            {"label": "wrong", "kind": "ir", "instruction": None,
             "tdi": TDI_IR, "tdo": padded}]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    g = _group(r.json(), "ir")
    wrong = next(x for x in g["runs"] if x["label"] == "wrong")
    assert wrong["status"] == "excluded"
    assert wrong["excluded_reason"] == "register_length_mismatch"
    assert wrong["offset_observed"] == 15
    diag = next(d for d in g["diagnoses"]
                if d["kind"] == "register_length_mismatch")
    assert diag["runs"] == ["wrong"]
    assert diag["register_length_expected"] == 10
    assert diag["first_anomaly"]["tdo_index"] == 15
    assert "TDO bit 15" in diag["detail"]
    # aligned runs still produce clean statistics
    assert g["summary"]["aligned"] == 2
    assert g["summary"]["unstable_bits"] == 0


def test_leading_missing_bits_reported_as_missing_interval(client):
    vid = _version(client)
    # echo starts 2 bits early: first two capture bits (DEVB) were truncated
    tdo = IR_CAP_SHIFT[2:] + TDI_IR[:len(TDI_IR) - 8]
    runs = [ir_run("g1", IR_CAP_SHIFT),
            ir_run("g2", IR_CAP_SHIFT),
            ir_run("trunc", tdo)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    g = _group(r.json(), "ir")
    trunc = next(x for x in g["runs"] if x["label"] == "trunc")
    assert trunc["status"] == "aligned"
    assert trunc["leading_missing"] == 2
    assert g["missing_intervals"]
    first = g["missing_intervals"][0]
    assert first["start_bit"] == 0 and first["end_bit"] == 1
    assert first["runs"] == ["trunc"]
    # the missing bits hold 2 observations, the rest 3
    assert len(g["bits"][0]["observed_runs"]) == 2
    assert len(g["bits"][2]["observed_runs"]) == 3


def test_groups_separated_by_kind_and_instruction(client):
    vid = _version(client)
    tdi_bp = "111000101100"
    bp = lambda label: {
        "label": label, "kind": "dr", "instruction": "bypass",
        "tdi": tdi_bp, "tdo": "00" + tdi_bp[:len(tdi_bp) - 2]}
    runs = [ir_run("i1", IR_CAP_SHIFT), ir_run("i2", IR_CAP_SHIFT),
            bp("b1"), bp("b2")]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    assert r.status_code == 201, r.text
    groups = r.json()["groups"]
    assert [(g["kind"], g["instruction"]) for g in groups] == [
        ("ir", None), ("dr", "BYPASS")]
    bp_group = _group(r.json(), "dr", "BYPASS")
    assert bp_group["register_length"] == 2
    assert [s["register"] for s in bp_group["segments"]] == ["DR:BYPASS"] * 2
    assert bp_group["summary"]["unstable_bits"] == 0


def test_idcode_intermittent_mapped_to_device(client):
    vid = _version(client)
    # longer TDI so the echo after the 33-bit capture region is verifiable
    tdi_id = "011011001010110010110010101100101100101011001011"

    def idc(label, flips=()):
        bits = list("0" + IDC_SHIFT)
        for f in flips:
            bits[f] = "1" if bits[f] == "0" else "0"
        cap = "".join(bits)
        return {"label": label, "kind": "dr", "instruction": "IDCODE",
                "tdi": tdi_id, "tdo": cap + tdi_id[:len(tdi_id) - len(cap)]}

    runs = [idc("a"), idc("b", flips=(6,)), idc("c"),
            idc("d", flips=(6,)), idc("e")]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    g = _group(r.json(), "dr", "IDCODE")
    # TDO bit 6: segment layout is DEVB bypass (bit 0) + DEVA 32 bits,
    # so chain bit 6 = DEVA IDCODE local bit 5
    diag = next(d for d in g["diagnoses"]
                if d["kind"] == "intermittent_device_toggle")
    assert diag["affected_devices"] == ["DEVA"]
    assert diag["affected_positions"] == [0]
    assert diag["register_bits"] == [6]
    assert diag["evidence"]["bits"][0]["register_bit"] == 5


def test_wire_mode_unknown_dr_instruction(client):
    vid = _version(client)
    # an instruction the saved version cannot lay out -> wire mode, no
    # device mapping but per-bit statistics and isolation still work
    tdi = "101010101010"

    def run(label, cap):
        return {"label": label, "kind": "dr", "instruction": "PRIVATE",
                "tdi": tdi, "tdo": cap + tdi[:len(tdi) - len(cap)]}

    runs = [run("a", "0101"), run("b", "0101"),
            run("c", "0111"), run("d", "0101")]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    assert r.status_code == 201, r.text
    g = _group(r.json(), "dr", "PRIVATE")
    assert g["mapping"] == "wire"
    assert g["register_length"] is None
    assert g["segments"] == []
    assert all(x["status"] == "aligned" for x in g["runs"])
    # capture region is the 4 bits before the echo
    assert all(x["capture_bits"] == 4 for x in g["runs"])
    diag = next(d for d in g["diagnoses"]
                if d["kind"] == "isolated_sample_anomaly")
    assert diag["runs"] == ["c"]
    assert diag["register_bits"] == [2]
    assert diag["affected_devices"] == []
    bit = g["bits"][2]
    assert bit["device"] is None and bit["register"] == "wire"
    assert bit["register_bit"] == 2


def test_alignment_failed_run_excluded(client):
    vid = _version(client)
    # severely truncated TDO (2 bits): no offset has >=4 comparable echo
    # bits, so the TDI echo cannot be verified at all
    garbage = "01"
    runs = [ir_run("g1", IR_CAP_SHIFT),
            {"label": "noecho", "kind": "ir", "instruction": None,
             "tdi": TDI_IR, "tdo": garbage},
            ir_run("g2", IR_CAP_SHIFT)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs})
    g = _group(r.json(), "ir")
    bad = next(x for x in g["runs"] if x["label"] == "noecho")
    assert bad["status"] == "excluded"
    assert bad["excluded_reason"] == "alignment_failed"
    diag = next(d for d in g["diagnoses"] if d["kind"] == "alignment_failed")
    assert diag["runs"] == ["noecho"]
    assert g["summary"]["aligned"] == 2
    for b in g["bits"]:
        assert "noecho" not in b["observed_runs"]


def test_batch_persisted_and_queryable(client):
    vid = _version(client)
    runs = [ir_run("r1", IR_CAP_SHIFT), ir_run("r2", IR_CAP_SHIFT)]
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid, "runs": runs, "note": "batch A"})
    bid = r.json()["batch_id"]

    lst = client.get("/consistency", params={"session": "cons"}).json()
    assert len(lst) == 1 and lst[0]["id"] == bid
    assert lst[0]["version_id"] == vid

    row = client.get(f"/consistency/{bid}").json()
    assert row["note"] == "batch A"
    assert [x["label"] for x in row["runs"]] == ["r1", "r2"]
    assert row["result"]["groups"][0]["register_length"] == 10
    assert client.get("/consistency/9999").status_code == 404
    # raw captures untouched: this session never went through /infer storage
    assert [c["session"] for c in client.get("/captures").json()] == ["cons"] * 3


def test_request_validation(client):
    vid = _version(client)
    # duplicate labels
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid,
        "runs": [ir_run("x", IR_CAP_SHIFT), ir_run("x", IR_CAP_SHIFT)]})
    assert r.status_code == 422
    # unknown version
    r = client.post("/consistency", json={
        "session": "cons", "version_id": vid + 100,
        "runs": [ir_run("x", IR_CAP_SHIFT)]})
    assert r.status_code == 404
    # session mismatch
    r = client.post("/consistency", json={
        "session": "other", "version_id": vid,
        "runs": [ir_run("x", IR_CAP_SHIFT)]})
    assert r.status_code == 422
