# Misclassification analysis — 2026-08-09 NBC/USA broadcast

Findings from the frames flagged via `/report_wrong` during the Iowa Corn 350
on USA Network (`classifier_profile = nascar_on_nbc`), plus the related June
flags from the Prime Video feed. Fifteen frames were flagged on 2026-08-09 in
three bursts; everything else in `frames/` predates the current classifier.

Verification used Qwen3-Omni (the classifier's own model) for the audio and
re-ran the live `nascar_on_nbc` pipeline against the flagged frames, three
passes each at the production temperature of 0.2.

## Summary of the three flagged bursts

| Time | What it actually was | Ground truth | Classifier now (image only) |
|---|---|---|---|
| 19:29:14–22 | NASCAR-branded cinematic commercial | `ad` | 3 ad / 2 content |
| 21:12:32–41 | Chevy/Shell-style motorsport montage commercial | `ad` | 3 ad / 2 content |
| 21:19:54–21:20:02 | Credit One Bank sponsored squeezeback, race live in a large inset | `content` | 5 ad / 0 content |

Per-frame the verdicts are stable (3/3 agreement on 14 of 15 frames), so the
flapping inside a burst is not sampling noise — consecutive shots of the same
commercial genuinely look different to the model. The classifier is right about
the *commercial*, wrong about individual *shots* within it.

## 1. Audio capture has been dead since early July

The most consequential finding, and it is not a classifier bug.

`config.json` sets `enable_llm_audio: true`, so both LLM passes receive a
4-second WAV. On 2026-08-09 **61% of those clips are digital silence** (peak
amplitude 0–1 out of 32767); all five frames in the 19:29 burst are byte-for-byte
identical silence. Sampling 25 clips per capture day:

| Day | clips | silent (peak ≤ 2) | median peak |
|---|---|---|---|
| 2026-06-21 | 4169 | 0/25 | 8624 |
| 2026-06-28 | 2635 | 0/25 | 8648 |
| 2026-07-05 | 1280 | 3/25 | 10 |
| 2026-07-13 | 1375 | 8/25 | 3 |
| 2026-07-19 | 425 | 16/25 | 1 |
| 2026-07-26 | 1309 | 12/25 | 3 |
| 2026-08-09 | 2320 | 16/25 | 1 |

Every day through 2026-06-28 is clean; it breaks from 2026-07-05 onward.

**Likely mechanism.** `native_host/audio_capture.py` binds the monitor source of
whatever sink is *default at launch*. `audio_capture.log` shows the auto-detected
source alternating between `alsa_output.pci-0000_67_00.1.hdmi-stereo.monitor` and
`bluez_output.50_C0_F0_CA_16_8D.1.monitor` across restarts. If the browser's
audio is on a different sink than the one the host bound — or the default sink
changes after the host starts — the monitor is real but silent, and nothing in
the pipeline notices.

**Why it is worse than no audio.** Asked to describe the silent clips, Qwen3-Omni
confidently invented play-by-play for all fifteen:

```json
{"speech": "play_by_play_commentary", "transcript": "He's got a great start!",
 "music_bed": false, "crowd_or_engine_noise": true,
 "verdict": "live_race_broadcast", "why": "Commentary and engine noise indicate a live race."}
```

`crowd_or_engine_noise: true` on a file whose maximum sample value is 1. Both
`_report_racing_related` and `classify_by_prompt` ask the model to weigh audio
alongside the image, so silence is being laundered into a positive "this is live
racing" signal. Re-running the flagged frames with and without the silent audio
shows it doesn't shift the final verdicts much, but it does destabilise the quick
check (`model_quick_reject` fires inconsistently across identical inputs).

Suggested fixes, in order of value:

- Pin the capture device with `AUDIO_DEVICE` instead of auto-detecting, or
  re-resolve the default sink on each `get_audio` rather than once at startup.
- Gate audio at the source: compute peak/RMS in `audio_capture.py` (or in
  `classify.py` before base64-encoding) and pass `audio_bytes=None` when the clip
  is effectively silent. A silent clip carries no information and this model will
  not admit that.
- Log or expose a metric for silent-clip rate so this fails loudly next time.

## 2. The NASCAR NON STOP side-by-side check works on NBC/USA but not on Prime

The `nascar_on_nbc` docstring says the side-by-side half is UNVALIDATED because
"no NBC or USA ad-break frame with the banner exists in the dataset". That is no
longer true — the 2026-08-09 archive contains **94 frames carrying the banner**,
across roughly 13 separate breaks, and the check fired on 93 of them
(`classification_reason: side_by_side`). It is validated, and the docstring
should say so.

Two caveats worth recording:

- **The margin is thin.** Real banner frames score 0.89–0.90 against a threshold
  of 0.80. `non_stop_full` contributes nothing (0.24); only `non_stop` carries
  the check.
- **It fails completely on the Prime Video feed.** The same graphic on the
  2026-06-01/07/14/21 "Watch …" broadcasts scores 0.16–0.22, because the white
  mask (`min_thresh=200`) shreds that feed's softer, anti-aliased rendering of
  the banner. Those June frames are exactly the ones flagged as wrong at the
  time.

A single **unmasked grayscale** template covers both feeds. Cropped from
`2026-06-21T21-04-26-980026_0.jpg` at `[33:75, 55:345]` in 1920×1080 space
(42×290 px) and matched with `TM_CCOEFF_NORMED` over the region
`x 0–500, y 0–160`:

| Set | n | Result |
|---|---|---|
| June Prime banner frames (crop source excluded) | 52 | min 0.398, **median 0.998**, 90.4% ≥ 0.80 |
| Aug 9 USA banner frames | 94 | min 0.892, median 0.955 |
| Labelled `content` frames | 239 | **max 0.656**, 0 above 0.80 |
| Aug 9 non-banner frames | 2226 | max 0.71 |

That is a ~0.18 margin on both feeds from one template, versus the current
0.09 margin on one feed and total failure on the other. The 10% of June
positives below threshold are the banner animating in and out.

## 3. Sponsored squeezebacks are the systematic error

The 21:19 burst is the only one the classifier gets wrong every single time, and
it is a layout the prompt has no rule for. Credit One Bank wraps the frame in an
L-bar — logo panel down the left, product bar across the bottom — while the live
race continues in a large inset. Measured edges: the inset starts at x≈521,
y 0–799, i.e. **54% of the screen area is uninterrupted live race**.

`prompt_nbc.txt` pushes this to `ad` from two directions —
"the race reduced to a small inset while the majority of the screen shows
unrelated content" and "a brand logo or slogan dominating most of the screen" —
and the model duly cites both. But the operator's call is `content`: the race is
still watchable, so there is nothing to switch away from.

The discriminator against a true NASCAR NON STOP break is size and furniture:

| | Live-race window | Leaderboard | NON STOP banner |
|---|---|---|---|
| NASCAR NON STOP break (`ad`) | ~15% of frame | yes | yes |
| Sponsored squeezeback (`content`) | ~54% of frame | no | no |

**The prompt route was tried and does not work.** `prompt_nbc.txt` now carries an
explicit squeezeback rule, stated three different ways across three iterations —
as a bullet in the RACE BROADCAST list, as a numbered decision procedure, and as
a structural "one video panel vs two" test. Measured over 8 interleaved reps on
the flagged frames (120 verdicts per arm), the squeezeback burst goes from
**0/40 to 5/40**. The model will even name the layout — one reply reads *"race
cars on track with credit one bank branding and a 'squeezeback' layout"* — and
still answer `type=ad`. Prominent branding dominates its judgement regardless of
what the prompt says about it.

An early size-based phrasing ("race window roughly half the screen or more →
racing") was actively harmful: the model read Fox side-by-side breaks with ~38%
race panels as "large" and flipped four labelled `ad` frames to `content`. The
current wording keys on whether a *second video panel* is playing, which is the
real structural difference and does not have that failure mode.

So this one needs code, not prompting:

- **Geometry check.** `rectangle_match.py` already has the machinery. The
  squeezeback inset is a stable normalized box — measured at x≈521, y 0–799,
  i.e. 54% of frame area, consistent across all five frames. A rule of "one
  detected video rectangle ≥ ~40% of frame area, no NON STOP banner, no second
  large rectangle → content" would generalise past any one sponsor, and unlike
  the prompt it would actually fire.

For reference, the sponsor logo itself is trivially matchable — a crop of the
Credit One panel at `[110:235, 90:470]` scores 0.994–0.997 on its own frames and
maxes at 0.367 over 600 unrelated frames — but it only appeared in one break
(14 frames) all day, so per-sponsor templates are poor value compared to the
geometric rule.

Note the corner bug cannot rescue this case: the squeezeback scales the video
down, so neither the peacock nor the USA wordmark is where `PEACOCK_REGION` /
`USA_REGION` look for them.

## 4. Cinematic racing commercials — the hard residual

The 19:29 and 21:12 bursts are commercials built entirely from real motorsport
footage, and per-shot the model is right about 60% of the time. What it gets
wrong are the shots that are, in isolation, indistinguishable from a broadcast
camera: a tyre close-up, a burnout, a victory-lane celebration.

Three signals visible in these frames that the prompt does not currently name:

- **Zero broadcast furniture.** No bug, no leaderboard, no lower third, no score
  strip, for several consecutive seconds. Live coverage almost never runs that
  long completely clean; ads always do. This is the strongest available cue and
  it is a *negative* one, which is why it needs stating explicitly.
- **Legal disclaimer microtext.** `2026-08-09T21-12-41-070676_4` carries
  "Available features shown throughout" in small type at bottom centre, and the
  21:12:41 model reply picked up "available in selected markets". Small legal
  text anywhere near the bottom edge is close to a guarantee of an ad. Worth an
  explicit prompt line, and potentially an OCR check.
- **Incoherent scene sequence.** The 21:12 burst runs an F1 car, a desert
  off-road truck, an Indianapolis pit-lane shot and a night-race burnout in nine
  seconds. No live broadcast cuts across series and venues that fast. The
  existing "any race cars other than NASCAR Cup Series" rule catches individual
  frames here, but the sequence itself is the signal.

The first two are now in `prompt_nbc.txt`. Measured over 8 interleaved reps on
the flagged frames, the revised prompt scores **53/120 against the old prompt's
42/120**; on a broader sample of 90 labelled frames that reach the LLM the two
are indistinguishable, and repeat runs of the *same* prompt on the same frames
vary by ±2, so treat that second measurement as a no-op rather than a win.

Beware how noisy this model is at the per-frame level. At the production
temperature of 0.2, single frames flip verdict between otherwise identical runs
— `2026-08-09T19-29-14` returned 0/8 `ad` under one prompt and 4/8 under
another, and 3/3 `content` then 3/3 `ad` then 4/5 `content` across three
three-rep runs. Any future prompt comparison needs interleaved arms and at
least ~100 verdicts per arm; three reps on fifteen frames measures nothing.

That last point is the general one. The classifier is per-frame and stateless,
and every failure in this analysis is easier at the burst level than at the
frame level. The rolling buffer in `frame_saver.py` already holds the recent
frames; a temporal smoother — majority vote or hysteresis over the last N
classifications — would have produced the correct answer for both `ad` bursts
even with the model's current per-shot accuracy, and would suppress the
switch-flapping the operator was reacting to when they hit the report button.
