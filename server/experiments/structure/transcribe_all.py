"""Transcribe the whole broadcast to a timestamped word/segment stream.

Runs in ~300 s chunks with a small overlap so a sentence spanning a chunk edge
is still recognised once in full; the overlap is trimmed on the way out by
dropping segments that start before the previous chunk ended.

CPU int8 on 20 threads runs about 2x realtime here, so the whole 2h40m takes
roughly 75 minutes. GPU would be far quicker but /dev/nvidia* is outside the
sandbox, so ctranslate2 reports no CUDA device.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audio_track import build, write_wav  # noqa: E402
from timeline import load  # noqa: E402

HERE = Path(__file__).parent
MODEL = Path.home() / "Code/projects/faster-whisper-py/models/faster-whisper-large-v3"

CHUNK = 150  # frames, ~300 s
OVERLAP = 5  # frames, ~10 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "transcript_full.jsonl"))
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--threads", type=int, default=20)
    args = ap.parse_args()

    sys.path.insert(0, str(Path.home() / "Code/projects/faster-whisper-py"))
    from faster_whisper import WhisperModel

    rows = load()
    model = WhisperModel(
        str(MODEL), device="cpu", compute_type="int8", cpu_threads=args.threads
    )
    tmp = Path(args.tmp) / "chunk.wav"

    done_until = -1e9
    started = time.time()
    with open(args.out, "w") as f:
        for i0 in range(0, len(rows), CHUNK):
            i1 = min(len(rows) - 1, i0 + CHUNK + OVERLAP - 1)
            pcm = build(rows, i0, i1)
            write_wav(tmp, pcm)
            # The track starts one full clip (~4 s) before frame i0's timestamp.
            t_start = rows[i0]["t"] - 4.0
            segments, _ = model.transcribe(
                str(tmp),
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
                vad_filter=True,
            )
            n = 0
            for s in segments:
                abs_s, abs_e = t_start + s.start, t_start + s.end
                if abs_s < done_until - 0.5:
                    continue
                f.write(
                    json.dumps(
                        {
                            "s": round(abs_s, 2),
                            "e": round(abs_e, 2),
                            "t": s.text.strip(),
                        }
                    )
                    + "\n"
                )
                n += 1
            done_until = rows[min(i1, i0 + CHUNK - 1)]["t"]
            f.flush()
            el = time.time() - started
            print(
                f"  frames {i0}-{i1}  {n} segs  t<={done_until:.0f}s  "
                f"elapsed {el / 60:.1f}min",
                file=sys.stderr,
                flush=True,
            )
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
