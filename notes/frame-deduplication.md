# Frame deduplication

Findings from measuring perceptual-hash deduplication against the full
`server/frames/` corpus (57,117 images), and the rationale behind the defaults
in `scripts/dedupe_frames.py`.

## Why the corpus is redundant

Two mechanisms produce near-identical frames:

- **Burst capture.** Flagging a frame as wrong saves the whole rolling buffer of
  recent frames. When the broadcast is static — a caution lap, a studio desk
  shot, a full-screen bumper — those frames are visually interchangeable.
- **Repeated ads.** The same commercial and the same network bumper recur across
  broadcasts. The "WE'LL BE RIGHT BACK" NASCAR card alone appears 286 times with
  a byte-for-byte identical perceptual hash, spread over at least five dates.

Roughly three quarters of removable frames come from the first mechanism
(same-day bursts) and a quarter from the second (cross-day repeats), so a
time-windowed heuristic would leave a substantial fraction of the win on the
table. Hashing catches both.

## Measured impact

Baseline: **7.16 GiB images + 14.07 GiB audio + 0.06 GiB thumbnails ≈ 21.3 GiB**.

Greedy keep-first grouping, protected frames excluded from removal:

| phash threshold | frames dropped | images | audio | total freed |
|---|---|---|---|---|
| 0 (identical hash) | 14,583 | 1.65 GiB | 3.82 GiB | 5.48 GiB |
| 4 | 17,894 | 2.05 GiB | 4.76 GiB | 6.82 GiB |
| 6 | 19,650 | 2.27 GiB | 5.24 GiB | 7.52 GiB |
| 8 | 21,586 | 2.52 GiB | 5.76 GiB | 8.29 GiB |
| **10 (default)** | **23,611 (41%)** | **2.78 GiB** | **6.29 GiB** | **9.08 GiB** |
| 12 | 25,749 | 3.05 GiB | 6.84 GiB | 9.90 GiB |

Byte-identical files (matching MD5) account for 12,456 of those frames — that
subset is risk-free regardless of threshold.

## Why threshold 10

The 459 frames manually labelled `ad` or `content` are ground truth: any pair
with differing labels must never be merged. Across all 105,111 such pairs:

- **phash alone merges zero conflicting pairs up to distance 11.** The first
  cross-label merge appears at 12, and precision degrades quickly after
  (96.7% at 16, 87.4% at 20).
- **Requiring phash *and* dhash to agree holds to 14**, with the first error at 15.

Threshold 10 therefore sits a clear margin below the first observed mistake,
and matches `AppConfig.phash_threshold`, which the override system already uses.
The dhash cross-check is kept anyway — it costs nothing and widens the margin.

The failures at distance 12–14 are instructive: they are *structurally* similar
rather than visually similar, typically two different split-screen ad breaks
sharing a scoreboard rail and the same block layout. A DCT hash of a 16:9
broadcast frame keys heavily on gross layout, so any threshold loose enough to
merge different layouts of the same scene is also loose enough to merge
different scenes with the same layout.

## Grouping must not be transitive

This is the one decision that will silently destroy the corpus if it is made
wrong. Treating "within threshold" as an equivalence relation and taking
connected components chains unrelated frames together through intermediates:

| threshold | largest component | max distance within it |
|---|---|---|
| 2 | 1,093 | 30 |
| 10 | 1,747 | 52 |
| 12 | 17,307 | 60 |
| 16 | 56,711 (of 57,117) | 62 |

At threshold 16 a single component holds 99.3% of the corpus. Frames 62 bits
apart — the practical maximum for a 64-bit hash — end up in the same group.

The script instead does a **greedy keep-first pass**: iterate in chronological
order and drop a frame only if it is within the threshold of a frame already
*kept*. Every removal is then justified by a direct comparison against the
survivor that replaces it, and no chain of intermediates can form.

Note that `scripts/find_dupes.py` predates this analysis. It is O(n²) in both
passes and its grouping marks `visited` inconsistently, so it is only usable on
a few hundred frames — not on the full save dir.

## Degenerate hashes

48 frames hash to `0000000000000000` and 78 to `0000000000000080`. These are
solid-black fade transitions, and they are what bridges otherwise unrelated
clusters, since anything near-uniform lands close to zero. They carry no
training value; `--drop-blank` removes them outright rather than treating one as
a representative worth keeping.

## Audio is a separate decision

Audio is 14.07 GiB against the images' 7.16 GiB, so **69% of the headline saving
is audio, not images**. But a frame's clip shares only its stem, not its
content: a static shot with different commentary is exactly the case where the
image is redundant and the audio is not. Deleting clips because their frames
collide discards genuinely distinct data, which matters while `enable_llm_audio`
is a live option.

Audio removal is therefore opt-in via `--include-audio`. Without it the script
frees 2.78 GiB at the default threshold and leaves `audio/` untouched.

## Protected frames

Anything carrying manual work is never deleted, and is preferred as the
representative its neighbours collapse into:

- a label in `labels.json`
- a feature record in `features.jsonl`

Ordering protected frames first in the greedy pass makes them representatives,
which is why the measured label loss is zero at every threshold. Metadata files
are rewritten to drop records for removed frames, so no entry is left dangling.

## An inconsistency this surfaced

Deduplication is a decent audit of the labels themselves. Two findings worth
knowing:

- `2026-03-11T20-27-07-118661_0.png` and `..._3.png` are **byte-identical**
  (same MD5) but labelled `content` and `ignore`.
- `2026-03-11T20-27-19-278136_3.png` (`ad`) and `2026-03-11T20-27-23-326460_4.png`
  (`content`) are near-identical UFL side-by-side ad-break frames four seconds
  apart, labelled oppositely.

`--report-conflicts` lists pairs like these without deleting anything.
