# Temporal smoothing and hysteresis — experiment

An attempt to answer whether the classifier should keep state between frames,
prompted by the closing paragraph of `misclassification-analysis-2026-08.md`:
*"a temporal smoother — majority vote or hysteresis over the last N
classifications — would have produced the correct answer for both `ad` bursts."*

The short version: **for the race itself that is no longer true, because there
is almost nothing left to smooth.** During green-flag racing the current
pipeline is right on 99.8% of frames, OpenCV settles 87% of them with zero
errors, and the two-frame debounce already in `receive.py` produces exactly the
right number of switches with no flapping at all. Every deeper temporal policy
tried here either matched it or made it worse.

That is a different answer from the one the August analysis expected, and the
reason is measurement, not disagreement — see "Why the archive says otherwise".

## What was measured

Two captures, because the obvious one turns out to be unusable for this
question.

**The burst archive** (`server/frames/`, 2026-08-09 Iowa Corn 350 on USA) is
what the earlier analysis used. `recent_frames` is a `deque(maxlen=5)` flushed
once a minute, so the archive is not a timeline: it is 208 disjoint ~8-second
bursts covering 23% of the broadcast. Nothing needing more than 8 seconds of
history can be simulated on it. Worse, bursts are written when the classifier
*changes its mind* (`suspicious_debounce`), so transitions are over-represented
by roughly 4x and the hard frames are over-represented by much more.

**The continuous recording** (`record_broadcast.py`, 1572 frames over 51
minutes) has a 2.0 s cadence and no gaps at all. This is the one the conclusions
rest on. Restricted to green flag onward it is 1264 frames / 42 minutes with 7
real transitions.

Ground truth was labelled by eye from contact sheets, never from the classifier.
On the continuous recording the two OpenCV checks anchor it — 48/48 sampled
network-bug frames verified as live coverage, 16/16 side-by-side frames as
breaks — and all 378 frames those checks leave undecided were reviewed
individually. On the burst archive the labels agree with the operator's own
`/report_wrong` corrections on 10/10 frames.

The pipeline was then re-run for real (`classify_image`, audio off, 3 reps) to
get per-frame verdicts and wall times, and every policy replays those.

## Where the error actually is

Race window, 3792 verdicts (1264 frames x 3 reps):

| Stage | verdicts | errors | error rate |
|---|---|---|---|
| OpenCV (bug / side-by-side) | 3306 | **0** | 0.0% |
| LLM fall-through | 486 | 7 | 1.4% |

OpenCV settles 87% of race frames in a median of 5 ms and did not get one wrong.
All seven LLM errors are the same direction — live content called `ad`.

Across the whole 51 minutes including the pre-race show the picture is worse
(96.5%), but that is almost entirely two produced montages:

| Segment | verdicts | accuracy |
|---|---|---|
| Everything except the two opening montages | 4440 | **99.55%** |
| The pre-race open and the race-open tease (130 s total) | 276 | **47.8%** |

Those two montages are cinematic, letterboxed, carry no network bug and no
leaderboard, and are cut from real racing footage — they match the prompt's own
"a COMPLETELY clean frame … is close to a guarantee of an ad" rule almost
perfectly. The classifier is at chance on them. They are also both in the
pre-race show, where auto-switching is not used, so they are out of scope; they
are recorded here only because they dominate any whole-broadcast average and
would otherwise look like a general accuracy problem.

## The policies

All replay the same per-frame verdicts, so differences are the policy alone.
Each starts from the ground-truth state, which is neutral across policies.

- `stateless` — emit each frame's verdict (what runs when debounce is off)
- `debounce2` — **current production**: OpenCV commits at once, an LLM verdict
  needs to repeat before it moves the state
- `majorityK` — majority of the last K verdicts
- `hystN/M[+cv]` — N consecutive `ad` to enter, M consecutive `content` to
  leave; `+cv` lets a direct OpenCV observation commit immediately
- `logodds` — weighted evidence accumulator with decay, weights by source
  (a side-by-side match counts 3x an LLM opinion)
- `bugmem` — hysteresis that also tracks how long since the network bug was last
  seen, demanding more evidence to call `ad` while the bug was on screen recently
- `cacheN+…` — skip the LLM when the frame is within N perceptual-hash distance
  of the previous one and reuse its verdict

### Race window (1264 frames, 42 min, 7 true transitions)

| policy | acc% | steady% | ad recall% | switches | flaps | llm% |
|---|---|---|---|---|---|---|
| stateless | 99.8 | 99.9 | 100.0 | 10 | 1.3 | 12.8 |
| **debounce2 (production)** | **99.4** | **100.0** | 98.1 | **7** | **0** | 12.8 |
| hyst2/2 | 99.4 | 100.0 | 98.1 | 7 | 0 | 12.8 |
| hyst3/2+cv | 99.3 | 100.0 | 97.2 | 7 | 0 | 12.8 |
| hyst4/2+cv | 99.1 | 100.0 | 96.3 | 7 | 0 | 12.8 |
| majority5 | 98.8 | 100.0 | 96.3 | 7 | 0 | 12.8 |
| logodds 2/1.5 | 97.5 | 98.9 | 90.7 | 7 | 0 | 12.8 |
| cache8+hyst3/2+cv | 99.3 | 100.0 | 97.2 | 7 | 0 | 11.3 |

`steady%` is accuracy away from a transition; `flaps` are switches reversed
within 8 s. Seven switches is exactly right — one per true transition.

Production's debounce hits 100% steady-state accuracy, zero flaps and the exact
switch count. Everything beyond it buys nothing and costs ad recall, because
every extra frame of evidence is another 2 s of commercial shown: hyst4/2 gives
up 1.8 points of ad recall for no measurable benefit. `logodds` is much too
sluggish — it is the right shape for a noisier classifier than this one.

### Burst archive (1339 frames, transition-heavy, includes pre-race)

The same policies on the archive rank completely differently:

| policy | acc% | steady% | boundary% | flaps |
|---|---|---|---|---|
| stateless | 93.8 | 93.6 | 96.7 | 52.0 |
| debounce2 | 92.1 | 93.6 | 64.8 | 21.0 |
| hyst3/3+cv | 91.5 | 94.2 | 43.2 | 5.3 |
| logodds 2/1.5 | 88.9 | 93.0 | 14.6 | 1.3 |

Here smoothing looks actively harmful — every policy scores below stateless.
Splitting accuracy by distance to a transition explains it: steady-state
accuracy is flat to slightly improved (93.6 → 94.2), while boundary accuracy
collapses (96.7 → 43.2). Lag costs everything at boundaries and earns everything
away from them, and this dataset is ~4x enriched in boundaries by construction.

### Why the archive says otherwise

The two datasets disagree because the archive is a biased sample of the same
broadcast, in three compounding ways:

1. It only keeps frames near a classifier disagreement, so it is dense in both
   transitions and genuinely hard frames. OpenCV settles 54% of archive frames
   against 87% of race frames.
2. Its 8-second bursts cannot contain a policy's warm-up, so a 3-frame
   hysteresis spends a third of every episode still catching up.
3. It predates `e076c43`, which reworked the NASCAR NON STOP banner to an
   unmasked template. That check now fires on 7% of frames with no false
   positives, moving work out of the LLM entirely.

The August analysis's conclusion was correct about the data it had. It does not
survive contact with a continuous capture of the current classifier.

## Latency

Budget was a median under 0.25 s and a maximum of 1.5–2 s.

| | median | p95 | p99 | max |
|---|---|---|---|---|
| Whole OpenCV pass (resize + 3 matches) | 2.4 ms | 3.8 ms | — | 11 ms |
| LLM quick check only (1 call) | 0.18 s | — | — | — |
| Quick check + full prompt (2 calls) | 0.50 s | — | — | — |
| **Per frame, race window** | **2 ms** | 0.47 s | 0.66 s | **4.97 s** |

The median is met with three orders of magnitude to spare, because OpenCV
settles 87% of frames. **The maximum is not met**: 8 of 1134 LLM calls exceeded
2 s, worst 4.97 s.

The obvious explanation — a rambling generation running to the 500-token cap —
is wrong. Re-running the slowest frames and measuring completion length puts it
at a median of 20 tokens and a maximum of 44, nowhere near the cap, and capping
`max_tokens` at 100 changed the verdict on 1 of 40 frames while leaving the
distribution essentially unmoved. So **capping `max_tokens` is not the fix**,
and the cause of the tail is still open: it is server-side variance rather than
anything about the request. Worth watching `/metrics`
(`classification_time_seconds` already has buckets out to 10 s) across a few
broadcasts before chasing it, since it is 0.2% of frames.

One cheap latency win remains untested for accuracy: dropping the quick check.
It is a second LLM call costing a median 0.18 s, and
`misclassification-analysis-2026-08.md` already found it fires inconsistently.
Removing it would take the two-call path from 0.50 s to ~0.32 s.

The near-duplicate cache is **safe but low-yield**. Over 1339 archive frames, no
pair within a perceptual-hash distance of 12 ever straddled a label change — so
reuse never introduces an error — but only 9% of frames fall within distance 4
and 19% within 12. Live video at a 2 s cadence simply is not that repetitive. In
the race window it takes LLM calls from 12.8% to 11.3% of frames. Worth having
for the p95 (0.47 s → 0.25 s) but it is not a lever on accuracy.

## Giving the model history directly

The policies above smooth the classifier's *output*. The other way to use
history is to change its *input*, so four arms were run live against
llama.cpp, interleaved per frame, on the burst archive — the continuous race
window has only 7 LLM errors in total, far too few to compare anything against.
Frames are the ones OpenCV does not settle, sampled at random:

| arm | acc% | ad recall% | content recall% | median latency |
|---|---|---|---|---|
| **base** (production: one image, `prompt_nbc.txt`) | **89.1** | 89.2 | 89.0 | 1.09 s |
| + textual context | 87.1 | 93.3 | 78.0 | 1.17 s |
| + two previous frames as images | 81.7 | 78.3 | 86.6 | 1.51 s |
| + both | 83.2 | 85.0 | 80.5 | 1.46 s |

(101 frames, 202 verdicts per arm. Absolute latencies are inflated — four arms
were hitting a two-slot server — but the comparison between them is fair.)

**Every form of context made it worse.** Handing the model the two preceding
frames is the worst single change measured anywhere in this experiment, costing
7.4 points; it appears to blend the images rather than classify the last one,
despite being told which to answer about, and it costs 40% more latency.

The textual arm is more interesting than its score. It did not degrade
uniformly — it moved ad recall up 4.1 points and content recall down 11.0. That
is the injected prior doing exactly what it was told: the context block asserts
that "a long stretch with no network bug at all is itself evidence of a break",
which is true in general (93.5% of such frames are ads) but false for precisely
the frames that are hard here, where bug-less live content is the failure mode.
A better-calibrated wording might come out ahead, but that is prompt tuning
against a known ±2 point noise floor, and the earlier analysis is emphatic about
how little such a difference would mean.

## Signals that did not pay off

**Broadcast-furniture absence.** The strongest single temporal feature measured:
on the archive, once the network bug has been gone more than 10 s the frame is
`ad` 93.5% of the time, against 0.5% when the bug is present. But it cannot be
exercised — 45% of archive frames sit in bursts too short (8 s) for the timer to
ever accumulate, and in the race window OpenCV has already settled anything the
bug would decide. `bugmem` scored identically to plain hysteresis everywhere.
It would also make the opening-montage failure *worse*, since a produced tease
legitimately runs a minute with no bug.

**USA logo in the lower right as an ad signal.** From the operator: the live bug
sits in the *upper* right, so a USA logo in the lower right is promo furniture.
Measured over the labelled race window with a multi-scale masked match, this is
perfectly precise — 0 false positives on 968 content frames at any threshold
≥ 0.55 — but only reaches 2.8% of ad frames, because most ads are third-party
spots with no USA logo at all. It is the only OpenCV check besides the banner
that would vote `ad`, and it costs about 1 ms, so it is cheap to add; it is just
not a lever.

## Recommendations

1. **Change nothing about the temporal policy for the race.** The existing
   two-frame debounce is already at the optimum on this evidence. Do not add
   hysteresis, majority voting or an evidence accumulator — all measured neutral
   to worse, and the sluggish ones give up real ad recall.
2. **Check that debounce is actually on wherever the server runs.**
   `AppConfig.enable_debounce` defaults to `False` while `AppState.enable_debounce`
   defaults to `True`, and startup unconditionally copies config over state. Docker
   is fine — `.env` sets `RECEIVER_ENABLE_DEBOUNCE=1` — but a dev server started
   without that variable silently runs the `stateless` policy, which flapped 26
   times in 51 minutes here. Making the two defaults agree would remove the trap.
3. **Do not feed the model previous frames.** Measured directly, it is the
   worst change tried here — see below.
4. **Consider dropping the LLM quick check** — a second call, 0.18 s, already
   known to be unstable. Needs an accuracy A/B before removing. This is the only
   latency lever identified; the per-frame maximum still misses the 2 s budget
   on 0.2% of frames for reasons not yet established.
5. **Enlarge `recent_frames`.** A 5-deep deque is why three years of archive
   cannot answer a question about temporal behaviour. Even 30 would make the
   saved bursts a minute long and let furniture-absence be evaluated properly.
6. **Keep recording continuous broadcasts.** One 42-minute window with 7
   transitions is not enough to characterise rare failures, and it is the only
   capture format that can measure any of this.

## What this does not establish

One broadcast, one network, 7 transitions. The race window contained no
sponsored squeezeback of the Credit One kind, which the August analysis
identified as the systematic error and which no temporal policy addresses. The
87% OpenCV coverage means most frames never exercise a decision at all, and
ground truth for those frames is anchored on the same checks that classify them
— verified by eye on 64 samples, but not independent. The pre-race montages are
a real and large failure that is simply out of scope rather than solved.
