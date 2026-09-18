"""Tests for scripts/annotate_broadcasts.py against a fake recording tree."""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from scripts import annotate_broadcasts as ab

HOST = "tv.youtube.com"


def make_broadcast(
    root: Path, name: str, n: int, *, audio_for=(), shuffle=False
) -> Path:
    """A recording directory with `n` frames, tiny JPEGs, and a manifest."""
    d = root / HOST / name
    (d / "images").mkdir(parents=True)
    (d / "audio").mkdir()
    records = []
    for i in range(n):
        fn = f"2026-08-13T02-05-{i:02d}-000000+00-00.jpg"
        Image.new("RGB", (64, 36), (i * 3 % 255, 0, 0)).save(d / "images" / fn)
        if i in audio_for:
            (d / "audio" / fn.replace(".jpg", ".wav")).write_bytes(b"RIFF")
        records.append(
            {
                "filename": fn,
                "timestamp": f"2026-08-13T02:05:{i:02d}.000000+00:00",
                "video_offset": 100.0 + 2 * i,
                "video_title": name.replace("_", " "),
                "network_name": "USA",
            }
        )
    if shuffle:
        records = records[::-1]
    with (d / ab.MANIFEST).open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return d


@pytest.fixture
def root(tmp_path: Path) -> Path:
    make_broadcast(tmp_path, "Iowa_Corn_350", 10, audio_for=(0, 1), shuffle=True)
    make_broadcast(tmp_path, "Cook_Out_400", 3)
    return tmp_path


@pytest.fixture
def library(root: Path) -> dict[str, ab.Broadcast]:
    return ab.scan_library(root)


@pytest.fixture
def iowa(library) -> ab.Broadcast:
    return library[f"{HOST}/Iowa_Corn_350"]


@pytest.fixture
def client(root: Path) -> TestClient:
    return TestClient(ab.app_factory(root))


def fn(i: int) -> str:
    return f"2026-08-13T02-05-{i:02d}-000000+00-00.jpg"


# ── Library and frames ───────────────────────────────────────────────────────


def test_scan_library_finds_recordings_by_relative_path(library):
    assert sorted(library) == [f"{HOST}/Cook_Out_400", f"{HOST}/Iowa_Corn_350"]
    bc = library[f"{HOST}/Iowa_Corn_350"]
    assert bc.title == "Iowa Corn 350"
    assert bc.network == "USA"
    assert bc.store.path == bc.root / "annotations.json"


def test_frames_sorted_by_timestamp_not_file_order(iowa):
    frames = iowa.frames()
    assert [f["filename"] for f in frames] == [fn(i) for i in range(10)]
    assert [f["i"] for f in frames] == list(range(10))
    assert frames[0]["has_audio"] and frames[1]["has_audio"]
    assert not frames[2]["has_audio"]
    assert frames[3]["t"] == 106.0


def test_position_tolerates_plus_decoded_as_space(iowa):
    assert iowa.position(fn(4)) == 4
    assert iowa.position(fn(4).replace("+", " ")) == 4
    assert iowa.position("nope.jpg") is None


def test_manifest_append_is_picked_up_without_reload(iowa):
    assert len(iowa.frames()) == 10
    with iowa.manifest.open("a") as fh:
        fh.write(
            json.dumps(
                {"filename": fn(10), "timestamp": "2026-08-13T02:05:10.000000+00:00"}
            )
            + "\n"
        )
        fh.write('{"filename": "half-written')  # a live recording mid-line
    frames = iowa.frames()
    assert len(frames) == 11
    assert frames[-1]["filename"] == fn(10)


# ── Records and store ────────────────────────────────────────────────────────


def test_validate_patch_accepts_fields_and_rejects_bad_values():
    ok = ab.validate_patch(
        {"verdict": "ad", "exclude": True, "note": "x", "video": "spot", "care_away": 3}
    )
    assert ok == {
        "verdict": "ad",
        "exclude": True,
        "note": "x",
        "video": "spot",
        "care_away": 3,
    }
    # Empty and null both clear; a false exclude is the same as none.
    assert ab.validate_patch({"note": "", "verdict": None, "exclude": False}) == {
        "note": None,
        "verdict": None,
        "exclude": None,
    }
    for bad in (
        {"verdict": "other"},
        {"video": "nonsense"},
        {"care_back": 4},
        {"care_back": True},
        {"nope": 1},
        {"exclude": "yes"},
    ):
        with pytest.raises(HTTPException):
            ab.validate_patch(bad)


def test_apply_note_modes():
    assert ab.apply_note("old", "new", "replace") == "new"
    assert ab.apply_note("old", "new", "fill") == "old"
    assert ab.apply_note("", "new", "fill") == "new"
    assert ab.apply_note("old", "new", "append") == "old || new"
    assert ab.apply_note("", "new", "append") == "new"
    assert ab.apply_note("old", "", "append") == "old"


def test_apply_patch_clears_with_none_and_prunes_empty():
    rec = ab.apply_patch(
        {"verdict": "ad", "note": "n"}, {"verdict": None, "video": "spot"}
    )
    assert rec == {"note": "n", "video": "spot"}
    assert ab.is_empty(ab.apply_patch(rec, {"note": None, "video": None}))
    assert not ab.is_empty({"exclude": True, "at": "now"})
    assert ab.is_empty({"at": "now"})


def test_store_round_trip_and_atomic_write(tmp_path: Path):
    store = ab.AnnotationStore(tmp_path / "annotations.json")
    assert store.all() == {}
    stored = store.put("a.jpg", {"verdict": "ad"})
    assert stored["verdict"] == "ad" and "at" in stored
    assert not list(tmp_path.glob("*.tmp"))
    data = json.loads((tmp_path / "annotations.json").read_text())
    assert data["schema"] == ab.SCHEMA_VERSION
    assert data["frames"]["a.jpg"]["verdict"] == "ad"

    # A second store on the same file sees the write; an outside edit is
    # picked up by the first one through the mtime check.
    other = ab.AnnotationStore(tmp_path / "annotations.json")
    assert other.get("a.jpg")["verdict"] == "ad"
    other.put("b.jpg", {"note": "hi"})
    assert store.get("b.jpg") == other.get("b.jpg")

    # Emptying a record removes it.
    assert store.put("a.jpg", {}) is None
    assert "a.jpg" not in store.all()


# ── Selection and fill ───────────────────────────────────────────────────────


def test_filters_and_boundary(iowa):
    iowa.store.put(fn(0), {"verdict": "content"})
    iowa.store.put(fn(1), {"verdict": "content"})
    iowa.store.put(fn(2), {"verdict": "ad", "risk": "fraught", "video": "spot"})
    iowa.store.put(fn(3), {"verdict": "ad", "audio": "ad_read", "video": "spot"})
    iowa.store.put(fn(5), {"exclude": True, "note": "black"})
    rows = ab.annotated_rows(iowa)
    boundary = [r["i"] for r, v in rows if r["boundary"]]
    # 1|2 is a verdict edge, 3|4 is the labeled/unlabeled frontier.
    assert boundary == [1, 2, 3, 4]
    counts = ab.counts_for(rows)
    assert counts["labeled"] == 4
    assert (
        counts["unlabeled"] == 5
    )  # 4, 6, 7, 8, 9 - the excluded frame is not "unlabeled"
    assert counts["ad"] == 2 and counts["content"] == 2 and counts["exclude"] == 1
    assert counts["noted"] == 1
    assert counts["fraught"] == 1 and counts["ad_read"] == 1
    assert counts["unfaceted"] == 3  # 0, 1 (no facets), 2 (no audio)
    sel, _ = ab.select(iowa, "ad")
    assert [r["i"] for r, v in sel] == [2, 3]
    with pytest.raises(HTTPException):
        ab.select(iowa, "conflict")


def test_fill_stops_at_previous_verdict_and_skips_excluded(iowa):
    iowa.store.put(fn(2), {"verdict": "content"})
    iowa.store.put(fn(4), {"exclude": True})
    iowa.store.put(fn(6), {"verdict": "content", "note": "keep me"})
    out = ab.fill_run(iowa, fn(6), "ad")
    assert out["first_i"] == 3 and out["last_i"] == 6
    assert out["filenames"] == [fn(3), fn(5), fn(6)]
    assert out["previous"] == {fn(3): None, fn(5): None, fn(6): "content"}
    store = iowa.store.all()
    assert store[fn(2)]["verdict"] == "content"
    assert store[fn(3)]["verdict"] == "ad"
    assert store[fn(4)] == {"exclude": True, "at": store[fn(4)]["at"]}
    assert store[fn(6)]["verdict"] == "ad" and store[fn(6)]["note"] == "keep me"


def test_fill_from_the_start_and_the_cap(iowa, monkeypatch):
    out = ab.fill_run(iowa, fn(2), "content")
    assert out["first_i"] == 0 and out["changed"] == 3
    monkeypatch.setattr(ab, "FILL_MAX", 3)
    with pytest.raises(HTTPException) as exc:
        ab.fill_run(iowa, fn(9), "ad")
    assert exc.value.status_code == 400
    assert iowa.store.get(fn(9)) is None


# ── Import ───────────────────────────────────────────────────────────────────


def test_convert_legacy_mapping():
    rec, kind = ab.convert_legacy(
        {"verdict": "ad", "judged": "ad", "note": "spot", "at": "x"},
        {
            "video": "spot",
            "audio": None,
            "risk": "safe",
            "care_away": 3,
            "care_back": 0,
            "phase": "race",
            "where": "in_break",
            "src": {"video": "derived"},
            "review": True,
        },
    )
    assert kind == "ad"
    assert rec == {
        "verdict": "ad",
        "note": "spot",
        "video": "spot",
        "risk": "safe",
        "care_away": 3,
        "care_back": 0,
    }
    rec, kind = ab.convert_legacy(
        {"verdict": "other", "note": "ad read"}, {"artifact": True}
    )
    assert kind == "other"
    assert rec == {"note": "ad read", "exclude": True}
    rec, kind = ab.convert_legacy(None, {"video": "crowd"})
    assert kind == "no_verdict" and rec == {"video": "crowd"}


def test_import_verdicts_dry_run_apply_and_overwrite(iowa, tmp_path: Path):
    verdicts = tmp_path / "v.json"
    facets = tmp_path / "f.json"
    verdicts.write_text(
        json.dumps(
            {
                "structure": {
                    fn(0): {"verdict": "content"},
                    fn(1): {"verdict": "other", "note": "hype"},
                    fn(2): {"verdict": "ad"},
                    "missing.jpg": {"verdict": "ad"},
                }
            }
        )
    )
    facets.write_text(
        json.dumps(
            {
                "schema": {},
                "facets": {
                    fn(2): {"video": "spot", "artifact": True},
                    fn(3): {"risk": "fraught"},
                },
            }
        )
    )
    iowa.store.put(fn(0), {"verdict": "ad", "note": "mine"})

    stats = ab.import_verdicts(iowa, verdicts, facets, "structure", False, False)
    assert stats["missing from manifest"] == 1
    assert stats["verdict other"] == 1
    assert stats["artifact -> exclude"] == 1
    assert stats["kept existing"] == 1
    assert stats["written"] == 3
    assert iowa.store.get(fn(2)) is None  # dry run wrote nothing

    ab.import_verdicts(iowa, verdicts, facets, "structure", False, True)
    store = iowa.store.all()
    assert store[fn(0)]["verdict"] == "ad"  # kept
    assert store[fn(1)] == {"note": "hype", "at": store[fn(1)]["at"]}
    assert store[fn(2)]["verdict"] == "ad" and store[fn(2)]["exclude"] is True
    assert store[fn(3)]["risk"] == "fraught"

    ab.import_verdicts(iowa, verdicts, facets, "structure", True, True)
    assert iowa.store.get(fn(0))["verdict"] == "content"


# ── API ──────────────────────────────────────────────────────────────────────


def test_index_and_broadcasts(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert '"live_race"' in r.text
    r = client.get("/api/broadcasts")
    rows = {b["id"]: b for b in r.json()}
    assert rows[f"{HOST}/Iowa_Corn_350"]["frames"] == 10
    assert rows[f"{HOST}/Cook_Out_400"]["labeled"] == 0


def test_frames_paging_and_errors(client: TestClient):
    b = f"{HOST}/Iowa_Corn_350"
    r = client.get("/api/frames", params={"b": b, "per_page": 4, "page": 3})
    d = r.json()
    assert d["pages"] == 3 and d["page"] == 3
    assert [x["i"] for x in d["rows"]] == [8, 9]
    assert d["counts"]["all"] == 10
    # Out-of-range pages clamp rather than 404.
    assert (
        client.get("/api/frames", params={"b": b, "per_page": 4, "page": 9}).json()[
            "page"
        ]
        == 3
    )
    assert (
        client.get("/api/frames", params={"b": b, "filter": "conflict"}).status_code
        == 400
    )
    assert client.get("/api/frames", params={"b": "nope/nope"}).status_code == 404


def test_annotate_bulk_and_sheet(client: TestClient):
    b = f"{HOST}/Iowa_Corn_350"
    r = client.post(
        "/api/annotate",
        json={"b": b, "filename": fn(1), "patch": {"verdict": "ad", "note": "x"}},
    )
    assert r.json()["rec"]["verdict"] == "ad"
    assert (
        client.post(
            "/api/annotate",
            json={"b": b, "filename": fn(1), "patch": {"verdict": "other"}},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/annotate", json={"b": b, "filename": "nope.jpg", "patch": {}}
        ).status_code
        == 404
    )

    r = client.post(
        "/api/annotate/bulk",
        json={
            "b": b,
            "filenames": [fn(1), fn(2), fn(3)],
            "patch": {"verdict": "content", "note": "run"},
            "note_mode": "append",
        },
    )
    assert r.json() == {"ok": True, "changed": 3, "cleared": 0}
    sheet = client.get("/api/sheet", params={"b": b, "filter": "content"}).json()
    assert [x["i"] for x in sheet["rows"]] == [1, 2, 3]
    assert sheet["rows"][0]["has_note"]
    page = client.get("/api/frames", params={"b": b, "filter": "content"}).json()
    assert page["rows"][0]["rec"]["note"] == "x || run"
    assert page["rows"][1]["rec"]["note"] == "run"

    r = client.post(
        "/api/annotate/bulk",
        json={
            "b": b,
            "filenames": [fn(2)],
            "patch": {"verdict": None, "note": None},
        },
    )
    assert r.json()["cleared"] == 1
    assert (
        client.post(
            "/api/annotate/bulk",
            json={"b": b, "filenames": [fn(2)], "note_mode": "smash", "patch": {}},
        ).status_code
        == 400
    )


def test_fill_and_context_routes(client: TestClient):
    b = f"{HOST}/Iowa_Corn_350"
    r = client.post("/api/fill", json={"b": b, "filename": fn(3), "verdict": "ad"})
    assert r.json()["changed"] == 4
    assert (
        client.post(
            "/api/fill", json={"b": b, "filename": fn(3), "verdict": "other"}
        ).status_code
        == 400
    )
    ctx = client.get(
        "/api/context", params={"b": b, "filename": fn(5), "radius": 2}
    ).json()
    assert [f["i"] for f in ctx["frames"]] == [3, 4, 5, 6, 7]
    assert ctx["pos"] == 2 and ctx["frames"][2]["current"]
    assert ctx["frames"][0]["verdict"] == "ad" and ctx["frames"][1]["verdict"] is None


def test_media_routes(client: TestClient, root: Path):
    b = f"{HOST}/Iowa_Corn_350"
    assert client.get(f"/image/{b}/{fn(0)}").status_code == 200
    assert client.get(f"/audio/{b}/{fn(0)}").status_code == 200
    assert client.get(f"/audio/{b}/{fn(5)}").status_code == 404
    assert client.get(f"/image/{HOST}/nope/{fn(0)}").status_code == 404
    # Starlette resolves the traversal before routing, so this never reaches
    # the guard; either way nothing outside images/ is served.
    assert client.get(f"/image/{b}/..%2F{ab.MANIFEST}").status_code in (400, 404)
    with pytest.raises(HTTPException):
        ab.safe_name("../classifications.jsonl")

    thumbs = root / HOST / "Iowa_Corn_350" / "thumbnails"
    assert not thumbs.exists()
    r = client.get(f"/thumb/{b}/{fn(0)}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    made = list(thumbs.glob("*.jpg"))
    assert len(made) == 1 and not list(thumbs.glob("*.tmp"))
    first = made[0].stat().st_mtime_ns
    client.get(f"/thumb/{b}/{fn(0)}")
    assert made[0].stat().st_mtime_ns == first  # served from cache, not regenerated
    assert client.get(f"/thumb/{b}/missing.jpg").status_code == 404


def test_reload_finds_new_recordings(client: TestClient, root: Path):
    make_broadcast(root, "New_Race", 2)
    assert len(client.get("/api/broadcasts").json()) == 2
    assert client.post("/api/reload").json() == {"broadcasts": 3}
    assert len(client.get("/api/broadcasts").json()) == 3
