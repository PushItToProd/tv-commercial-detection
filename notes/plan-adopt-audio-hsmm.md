# Adopting the audio sensor and the `hsmm` policy

## What this is

`notes/broadcast-structure-2026-08.md` left two recommendations unimplemented: use
the audio as a sensor, and replace the 2-frame debounce with the duration-aware
state machine. This plan is how those reach the live detector without betting the
matrix on a number measured once.

The short version: **the policy is not the risky part — the evidence behind it
is.** So the work is staged so that the risky part is measured before anything
switches an HDMI input on its say-so.

## What the evidence actually says

From `experiments/structure/results_manual.json`, scored against the manually
verified Iowa ground truth (`exclude` mode, so the 124 `other` frames are outside
the scoring mask), sorted by the experiment's cost function:

| evidence \| policy | acc | switches | flaps | ad_shown | race_missed | lat_med | missed tr. | cost |
|---|---|---|---|---|---|---|---|---|
| opencv+audio+furniture \| stateless | 0.985 | 68 | **23** | 62 s | 76 s | 0.0 | 0 | 262 |
| opencv+audio \| stateless | 0.984 | 64 | 22 | 78 s | 70 s | 0.0 | 0 | 304 |
| **opencv+audio \| hsmm** | **0.987** | **34** | **0** | 96 s | 26 s | 1.0 | 0 | 314 |
| opencv+audio+furniture \| debounce2 | 0.982 | 46 | 5 | 84 s | 82 s | 1.0 | 0 | 334 |
| opencv+audio+furniture \| hsmm | 0.978 | 34 | 0 | 68 s | 132 s | 1.0 | 0 | 336 |
| opencv+furniture \| stateless | 0.974 | 66 | 17 | 94 s | 142 s | 0.0 | 0 | 424 |
| opencv only \| hsmm | 0.821 | 28 | 0 | 1292 s | 364 s | 7.0 | 6 | 4240 |
| opencv only \| debounce2 | 0.830 | 12 | 0 | 1568 s | 2 s | 0.0 | 11 | 4706 |

The truth is 35 segments — **34 transitions**. `opencv+audio | hsmm` produces
exactly 34 switches with zero flaps: it recovers the true segmentation precisely,
leaving 96 s of commercial on screen across seventeen breaks (~5.6 s per break)
and switching away from 26 s of racing across the whole broadcast.

The rows above it score a lower *cost* while taking 64–68 switches with 22–23
flaps — roughly double the real transitions, with reversals inside 8 s. The cost
function weights ad 3:1 against race and **does not price flaps at all**, which is
why it ranks a thrashing policy first. For a thing driving a physical relay in
front of a television that ranking is wrong, and the flap column is the answer.

`hsmm` is also already the reasoning we want. Its docstring: *"a 'this looks like
content' vote arriving 20 s into a commercial break is not weak evidence to be
averaged in, it is evidence about something that has never once happened, and the
right thing to do with it is discard it."* That is what absorbs a single-frame
sensor error — an F1 car in a spot read as racing, a wrecked-driver interview
read as an ad. The duration data is why it works: NON STOP breaks measure
120–122 s (sd 0.7), full breaks 118–217 s, and the shortest interior content run
in the whole broadcast is 94 s.

## What the evidence does not say

Four things, and each one shapes a stage below.

**1. The baseline in that table is not the current system.** `evidence.py` builds
sensor sets from opencv anchors, furniture and audio. **There is no LLM sensor
anywhere in it.** The production pipeline is opencv anchors *plus* an LLM
fallback on every unanchored frame, so the `opencv only` rows — 11 missed
transitions, 1568 s of ad shown — describe a system nobody runs. The current
detector's real performance in these terms has never been measured. Nothing here
can be justified as "better than what we have" until it is.

**2. Out-of-fold is not held-out.** The audio p(ad) comes from an L2 logistic
regression scored by segment-blocked cross-validation, which correctly stops a
frame being scored by a model that saw its own segment. But every fold shares one
broadcast: the same production truck, the same commentators, the same audio
mastering chain, the same network. Cross-broadcast transfer is entirely
unmeasured, and the model was fit on NBC/USA while the next broadcast to run is
Fox.

**3. The policy needs a sensor floor.** `opencv only | hsmm` is *worse* than
debounce on the failure that matters — 6 missed transitions and 7 s median
latency. The state machine's dwell floors turn into blindness when the evidence
underneath is thin. Whatever ships must not fall back to `hsmm` on anchors alone.

**4. Audio capture is a known-silent failure mode.** An entire summer of
collection was lost to a monitor source the browser wasn't playing to — clips
arriving at the right length and cadence, full of zeros. `audio_health.py` now
detects it. Once a *decision* depends on audio, that detector stops being a data
hygiene tool and becomes a safety interlock.

---

## Stage 0 — measure the real baseline

Before anything is adopted, score the pipeline that actually runs today.

Replay the current `nascar_on_nbc` profile over all 4,775 labeled Iowa frames —
opencv anchors first, LLM fallback on the rest — and record each frame's
`ClassificationResult` (`source`, `type`, `reason`). Then score it through the
same `evaluate.score` used for the table above, under both `debounce2` and the
live route's actual rule (which is *not* `debounce2`: at
`routes/receive.py:136`, `result_source == "opencv"` bypasses debounce entirely,
and `unknown` results are skipped rather than breaking a run).

This is a few thousand local llama.cpp calls, run once, offline. It produces the
number every later decision is measured against, and it will very likely also
surface how much of the LLM's contribution is on unanchored frames — which is
exactly where the audio sensor is supposed to help.

Deliverable: an `llm` evidence column added to the lab's signal set, and a
baseline row in the same units as the table above.

## Stage 1 — shadow mode

Ship the whole policy path, wired to nothing.

Every `/receive` call computes, in parallel with the existing decision:

- an **evidence row** — `banner`, `bug`, `black`, `anchor`, and (once Stage 2
  lands) `p_audio`;
- the `hsmm` state that row implies;
- the existing debounce decision, unchanged, which is the only one that moves the
  matrix.

Both are logged to `shadow.jsonl` alongside the frame filename and offset, and
both are reported in `/is_ad/status` and rendered on the `/is_ad` page. The
operator watching the race sees a second opinion and can tell at a glance when
the two disagree.

This is the highest-value stage and it needs no labels at all. It runs on every
broadcast watched from now on, across networks, and produces exactly the
cross-broadcast evidence item 2 above says is missing. It also makes
`POST /report_wrong` far more valuable: a wrong-report during shadow mode is a
timestamped human ruling at precisely the moment the two policies were most
likely to differ — free boundary labels at the moments that matter.

Requirements:

- **Classifiers must expose their raw scores.** `hsmm` needs `banner` and `bug`
  separately, not collapsed into a verdict; `ClassificationResult` currently
  carries only `source` / `type` / `reason`. Add a `signals: dict` field that
  profiles populate (`peacock`, `usa`, `sbs` for NBC; the Fox equivalents plus
  `rectangle_match`), plus `black_frac`, which is cheap and which `hsmm` uses as
  a hard entry rule.
- **Dwell must be measured in seconds, not frames.** `policies.py` counts frames
  at an assumed 2 s cadence (`NONSTOP_LEN=60`, `MIN_AD_DWELL=50`,
  `MIN_CONTENT_DWELL=40`). Live, frames arrive irregularly, the capture interval
  is configurable in the extension popup, and the stream pauses. Convert the
  constants to seconds (120 / 100 / 80) and accumulate dwell from `video_offset`
  deltas.
- **Discontinuities reset the policy.** On `is_seeking`, on a backward
  `video_offset` step, or on a forward jump beyond a few times the nominal
  cadence, reset state rather than carrying a dwell across a discontinuity. The
  Icelandic capture already contains a 10 s backward seek, so this is not
  hypothetical. While `is_paused`, freeze dwell instead of accumulating it.
- **Restarts are honest about what they lost.** `AppState` is not persisted.
  `hsmm` initialises to `content` with effectively infinite dwell, so a restart
  mid-break starts in the wrong state and immediately switchable. Log it; don't
  hide it.

Shadow mode is worth keeping permanently, not deleting at promotion — it is how
any future policy change gets evaluated.

## Stage 2 — the audio sensor in the live path

The audio evidence is a trained model, and that is the part with real moving
pieces.

**The artifact.** `experiments/structure/audio_probe.py` fits plain L2 logistic
regression with hand-rolled gradient descent — no sklearn, so nothing new to
depend on. Ship the fitted model as a versioned JSON file next to the profile:

```json
{"version": 1, "features": [...21 names...], "win": 15,
 "mu": [...], "sd": [...], "w": [...85 floats...],
 "trained_on": {"program": "1LaATJR0CeM", "network": "USA", "frames": 4775},
 "metrics": {"blocked_cv_auc": ..., "note": "out-of-fold within one broadcast"}}
```

The provenance block is not decoration. It is what stops a model fit on USA
audio being quietly trusted on Fox.

**The features.** 21 per-clip DSP features (`rms_db`, `crest`, `dyn_range`,
`flatness`, `stationarity`, spectral flux, seven band ratios, …) over the 4 s WAV
already arriving on every `/receive`, then `_trailing(win=15)`: trailing mean and
sd over ~30 s, the current frame, and its difference from the trailing mean — 84
dimensions. The difference term is the change detector, and is why the sensor is
not 30 s late off the mark at a break's first frame.

**The state.** `AppState` gains a bounded deque of the last 15 feature vectors.
Below 15 the sensor reports no opinion rather than a bad one, so the first ~30 s
after a restart runs on anchors alone.

**Cost.** This is a scipy spectrogram over 4 s at 22.05 kHz on the request path,
every 2 s. Expected to be a few milliseconds; **measure it** and log a warning
above a threshold rather than assuming. `scipy` becomes a real dependency (today
it arrives only transitively via `imagehash`).

**Degradation is the safety-critical part.** A ladder, most to least evidence:

| condition | behaviour |
|---|---|
| audio healthy, ≥15 clips buffered | `hsmm` on opencv + audio |
| audio dead (`audio_health.is_silent`), or <15 clips, or no native host | **fall back to the current debounce path** — never `hsmm` on anchors alone |
| model version unknown, or profile has no model | current debounce path |

The middle row is item 3 above: `opencv only | hsmm` misses 6 of 34 transitions.
Falling back to the *policy* while losing the *sensor* is the one combination
measurably worse than doing nothing. The fallback must be visible in
`/is_ad/status` and on the page, next to the existing `audio_warning` banner.

## Stage 3 — promotion

`hsmm` moves the matrix only behind an explicit setting
(`POST /settings/policy`, values `debounce` | `hsmm`, default `debounce`),
and only once all of these hold:

1. **A held-out broadcast exists.** The Fox capture is labeled through the lab
   (`notes/plan-reusable-broadcast-lab.md`), and `opencv+audio | hsmm` is scored
   on it with a model that never saw it. Leave-one-program-out, not
   within-broadcast CV.
2. **It beats the Stage 0 baseline** on the metrics that matter: switch count
   within ~10% of the true transition count, **zero flaps**, median onset latency
   no worse, and no regression in `race_missed`.
3. **Shadow data agrees.** Across at least three shadow-mode broadcasts spanning
   at least two networks, the disagreements between shadow and live are reviewed
   and the shadow policy is right more often than not at the moments the operator
   pressed *report wrong*.
4. **The degradation ladder is exercised**, including a deliberate kill of the
   native host mid-broadcast.

If (1) shows the audio model does not transfer across networks — a real
possibility, since it was fit on one production's mastering chain — the answer is
a per-profile model, and the lab is where those get fit. That is a result, not a
failure: it would be the first honest measurement of a question the notes have
only ever gestured at.

## Failure modes worth naming up front

- **A model fit on 4,775 frames of one race, generalising to a sport it has
  never heard.** Road courses, rain delays, and a red flag all sound unlike Iowa.
  The dwell floors give some protection (a 100 s minimum ad dwell absorbs a lot
  of sensor noise), but a red flag is a long stretch of not-racing audio during
  actual content, and it is exactly the case `hsmm` would confidently call an ad.
- **NASCAR NON STOP handling is Fox/NBC-specific.** `hsmm`'s forced exit at
  `nonstop_len` assumes the banner marks a break of known fixed length. HBO Max
  has no traditional breaks at all and its side-by-side card means *content*, the
  opposite of the NBC banner. The policy's constants belong per-profile, not
  global.
- **The interaction between `phash_override` and a stateful policy.** An override
  short-circuits classification entirely today. Under `hsmm` it should enter as a
  strong anchor, not as a decision, or a single overridden frame can force a
  switch inside a dwell floor.
- **Nothing here changes what happens when the operator disagrees.**
  `report_wrong` currently sets `state.classification` and pauses auto-switch for
  30 s. Under `hsmm` it must also reset policy state, or the machine will spend
  its dwell floor arguing with the human.

## Verification

```bash
cd server

# Stage 0 — the missing baseline
uv run python scripts/replay_profile.py --program 1LaATJR0CeM --profile nascar_on_nbc
uv run python -m tv_commercial_detector.lab.eval --baseline
# expect: the live route's actual rule scored in the same units as
# results_manual.json, for the first time

# Stage 1 — shadow mode, no matrix movement
uv run pytest tests/test_policy_shadow.py
# assert: matrix.apply is never called from the shadow path;
# seek/pause/restart each reset or freeze dwell as specified
uv run uvicorn tv_commercial_detector.main:create_app --factory --port 11679
# then watch a broadcast and confirm both opinions render on /is_ad

# Stage 2 — the sensor
uv run pytest tests/test_audio_sensor.py
# assert: feature vector matches experiments/structure/extract_audio.py to 1e-9
#         on a fixture clip; <15 clips reports no opinion;
#         audio_health.is_silent forces the debounce fallback
uv run python scripts/bench_audio_features.py   # per-clip cost, must be « 2 s

# Stage 3 — held out
uv run lab eval --program XyIC6SWbJK0 --allow-test
```

## Out of scope

Fitting per-profile audio models, and any change to the classifier profiles'
OpenCV passes. Both wait on the lab having more than one labeled broadcast to
speak from.
