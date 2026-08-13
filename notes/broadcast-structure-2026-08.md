# Broadcast structure, audio, and a state machine that beats debounce

A follow-on to `temporal-hysteresis-2026-08.md`, which concluded that the
two-frame debounce was already at the optimum and that no deeper temporal policy
helped. That conclusion was right about the evidence it had. It does not survive
a longer capture, and the reason is that the earlier work smoothed the
classifier's *output* without ever giving the classifier a second sensor or a
model of the broadcast.

Three results, in order of how much they matter:

1. **Audio alone classifies this broadcast at AUC 0.97**, it is right on 100% of
   the ten-minute stretch where the OpenCV checks are completely blind, and
   trained on USA/NBC it still scores **AUC 0.915 on a Fox/FS1 broadcast** —
   within a whisker of a model trained on FS1 itself. This is the largest single
   improvement available, it needs no LLM, and it is the one signal here with
   evidence that it transfers.
2. **A duration-aware state machine produces exactly the right number of
   switches with zero flaps**, where a 2-frame debounce produces 48 switches
   with 8 flaps on the same evidence. The win comes from *one* asymmetric
   constraint, not from the elaborate duration model.
3. **The USA network bug is not a reliable content anchor.** NBC's own
   going-to-break bumpers carry it, at match scores of 0.91-0.99.

---

## What was measured

`/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/USA_4K_Iowa_Corn_350`
— the 2026-08-13 Iowa Corn 350 on USA, captured by `record_broadcast.py`.

| | |
|---|---|
| frames | 4775 (the recording kept running; it stood at 5974 by morning) |
| span | 2:39:35, cadence 2.01 s, **no gaps at all** |
| audio | 4 s clip per frame, all 4775 present, none silent |
| ad | 41.8 min, 26.3% of the broadcast |
| structure | 35 segments, 17 breaks, 34 transitions |

This is 3x the length of the continuous capture the hysteresis work used, and it
covers a full race rather than 51 minutes of it.

### Ground truth

Segment-level, and it took three passes because the first two were not good
enough:

1. Contact sheets over the whole broadcast at one frame in twelve, read by eye.
   This finds every break but places its edges only to about +/-30 frames.
2. Boundaries proposed independently from the furniture signal
   (`refit_truth.py`), which is precise at edges but fragments a break whenever
   a spot holds a static title card.
3. **Every frame where those two disagreed — 386 frames in 31 runs — was
   reviewed individually**, plus a final single-frame pass over the eight
   boundaries still ambiguous after that.

The two methods fail in opposite directions, which is what makes the
disagreement set worth reviewing: the eyeball pass was late at edges, the
furniture pass was fooled by static ad graphics. Neither certifies itself.

Two conventions had to be fixed to make edges well defined, and both are
judgement calls worth disagreeing with:

- **The full-screen NASCAR wipe belongs to the break.** This makes all six NON
  STOP breaks exactly 60 frames.
- **Live track or crowd carrying only a sponsor card is content** — the "IOWA
  CORN" aerial and the Progressive fan-cam at a rejoin. A produced sponsor spot
  with actors and no live footage is `ad` even when the sponsor is the race's own
  title sponsor, which is what the Iowa Corn commercial closing the 0:55:33 break
  is (frames 1742-1753, cornfields and actors, immediately before the rejoin).

---

## The shape of the broadcast

### Two kinds of break, and they are not alike

| | n | durations (s) |
|---|---|---|
| **NASCAR NON STOP** (side-by-side) | 6 | 120, 120, 120, 120, 120, 122 |
| **full commercial break** | 11 | 118, 130, 138, 156, 160, 160, 166, 178, 180, 184, 216 |

The NON STOP breaks are **120 seconds, standard deviation 0.7 s**. Every one is
58-59 frames of banner bracketed by a NASCAR wipe. This is not a distribution,
it is a constant.

Full breaks vary but have a hard floor: **the shortest is 118 s**. Content runs
between breaks have a floor too — the shortest interior run is 94 s. One break
every 9.4 minutes; closest two break *starts* are 261 s apart.

Break order was `F F F N N F F F F N F F N F F N N`, so the two kinds interleave
without an obvious pattern and cannot be predicted from position.

### A fourth presentation mode the classifier does not know about

Between 1:47:04 and 1:55:52 — 293 frames, nearly ten minutes — the broadcast
shows full-screen live racing with the running-order pylon down the left side
and **no corner bug at all**. The OpenCV anchor fires on **0 of those 293
frames**. Every one currently falls through to the LLM, and this is the material
`_report_racing_related` is documented as being weakest on.

So the broadcast has four modes, not the two the profile models:

| mode | furniture | current verdict |
|---|---|---|
| full-screen live | corner bug, upper right | `content`, OpenCV |
| NASCAR NON STOP | banner upper left, pylon, ad centre | `ad`, OpenCV |
| full commercial break | none | LLM |
| **pylon mode: live racing, no bug** | pylon left only | **LLM — and it is the hard case** |

### Broadcast phase buys nothing inside the race

The state machine was pictured as "something that knows which part of the
broadcast it's in", so: ad load by 20-minute block runs 19.6, 38.8, 26.1, 21.4,
20.6, 28.1, 20.1% with two or three breaks in each. Inside the race it is
**flat**. There is no denser stretch to anticipate, no visible "more ads under
caution" effect at this resolution, and no phase a policy could usefully
condition on.

The one real phase difference is the pre-race show:

| | span | ad load | breaks |
|---|---|---|---|
| pre-race | 10:03 | **65.4%** | one per 5.0 min |
| race | 2:29:30 | 23.6% | one per 10.0 min |

The pre-race show is two-thirds commercial, and it is also where the classifier
is weakest (the produced montages the previous notes measured at chance). But
auto-switching is not used there, so it stays out of scope.

So the useful state is *how long the current segment has run*, not *where in the
broadcast we are*. That is what the dwell floor encodes, and it is the whole of
what the structure turned out to be worth.

### Black frames

17 near-black frames in the whole broadcast, and **all 17 are inside ad breaks**
— perfect precision, but only 1.4% recall, and only 3 of them sit near a segment
edge. They are the joins between individual spots, not the break's boundaries.
Cheap to use as a hard `ad` signal; useless as a boundary detector.

The spot grid is much more visible in audio: near-silence recurs inside breaks
at roughly 15-30 s intervals (`analyse_structure.py` prints the pattern per
break), which is the 15/30/60 s spot boundary structure.

---

## Audio

The headline. Features are cheap DSP over each 4 s clip — band energies,
loudness dynamics, spectral flatness, within-clip stationarity — with no model
and no network call. `extract_audio.py` computes all 4775 in about a minute.

**Two things make a naive answer here wrong**, and both changed the result:

- Clips are 4 s and arrive every 2 s, so consecutive rows share half their
  samples. Splitting at random tests on audio already trained on: it inflates
  accuracy by about 2 points of AUC. Every split below is **by segment**.
- A single 4 s clip is the wrong unit. A break lasts at least 118 s, so the
  useful question is whether the last half-minute sounded like ads. The
  trailing-window model is worth about 1 point of AUC and is strictly causal.

| | AUC | acc |
|---|---|---|
| instantaneous 4 s clip | 0.959 | 0.922 |
| **trailing 30 s window** | **0.969** | **0.937** |
| (same, random split — leaky, for comparison) | 0.987 | 0.956 |
| on the 1415 frames OpenCV cannot settle | 0.936 | 0.882 |

Strongest single features: energy in **400-800 Hz** (AUC 0.882 for content) and
**60-150 Hz** (0.871 for ad). That is the engine-roar band against the
bass-heavy mastering of national spots — a physically sensible discriminator
rather than a coincidence, which is part of why it is worth trusting.

### It survives a temporal holdout

The same commercials re-air across breaks, so segment-blocking alone could still
be memorising spots. Training on one half and testing on the other:

| | AUC | acc |
|---|---|---|
| first half → second | 0.989 | 0.969 |
| second half → first | 0.965 | 0.940 |
| worst held-out quarter | 0.967 | 0.927 |

### And it is right exactly where vision is blind

On the 293-frame pylon stretch — no bug, no banner, no OpenCV verdict — **audio
calls `ad` on 0.0% of frames.** It is completely correct on the single hardest
region in the broadcast. The furniture signal gets 7.2% of it wrong.

---

## The state machine

All rows below use the same evidence — the existing OpenCV anchors plus the
audio sensor — so the differences are the policy alone. `ad_shown` is seconds of
commercial left on screen; `race_missed` is seconds of racing switched away
from. There are 34 true transitions.

| policy | acc% | steady% | switches | flaps | ad_shown | race_missed |
|---|---|---|---|---|---|---|
| stateless | 96.63 | 98.27 | 80 | 35 | 240 s | 82 s |
| debounce2 (production) | 96.57 | 98.57 | 48 | 8 | 232 s | 96 s |
| debounce8 (matched switch count) | 95.10 | 97.96 | 34 | 0 | 310 s | 158 s |
| **duration-aware state machine** | **97.03** | **98.91** | **34** | **0** | 242 s | **42 s** |

Read that table carefully, because the win is not uniform:

- **Against `debounce8`**, the only debounce setting that reaches the correct
  switch count, the state machine is 1.9 points more accurate, leaves 68 s less
  commercial on screen and misses 116 s less racing. Straightforwardly better.
- **Against production's `debounce2`**, it trades: 48 switches and 8 flaps become
  34 and 0, and `race_missed` drops from 96 s to 42 s, but `ad_shown` is *10 s
  worse* (242 s against 232 s). Debounce2 is quicker into a break and pays for it
  by thrashing the relay 14 extra times and cutting away from 54 s more racing.

So the honest claim is not "strictly better on every axis". It is that the state
machine is the only policy that hits the right switch count without flapping,
and it gets there without the large latency penalty that buying the same
stability from a longer debounce costs.

For reference, the same policies on OpenCV alone — no audio, no LLM — score 81%,
leave **30 minutes** of commercial on screen and never follow 11 of the 34
transitions. In production the LLM is what covers that gap. The audio sensor
covers most of it instead, at **6.8 ms of single-threaded DSP per clip** against
a measured 0.18-0.50 s for the LLM path, and without a network call.

### What actually does the work

This is the part that surprised me, and it makes the recommendation much simpler
than "build an HSMM":

| ablation | acc% | switches | flaps | race_missed |
|---|---|---|---|---|
| full state machine | 97.03 | 34 | 0 | 42 s |
| — without the **content** dwell floor | 92.75 | 42 | 2 | 450 s |
| — without the **ad** dwell floor | 96.82 | 34 | 0 | 36 s |
| — without the NON STOP forced timer | 97.03 | 34 | 0 | 32 s |
| — floor kept, accumulator disabled | 93.51 | 42 | 2 | 22 s |

**Only two ingredients matter**: a minimum dwell in `content` (~80 s), and
accumulating evidence instead of counting frames. The ad-side floor is nearly
free but nearly pointless, and the deterministic 120 s NON STOP timer — the most
striking regularity in the whole broadcast — **buys literally nothing**, because
the banner check already covers the entire break. I built it before I measured
it; it should not ship.

The asymmetry makes sense in hindsight. Once inside a break the audio sensor is
confidently `ad`, so there is nothing to suppress. During racing it occasionally
fires a false `ad`, and *that* is what needs a structural veto.

Parameters sit on a plateau, not a spike: any content floor from 30-46 frames
gives 34 switches and 0 flaps, and any CUSUM threshold from 0.5 to 6.0 nats does
too. The floors are deliberately conservative — 80 s against an observed
94 s minimum, 100 s against 118 s.

### Where the remaining error is, and why

Latency is now entirely at **full-break onsets**. NON STOP breaks switch in 2 s
in both directions because the banner is unambiguous. Full-break onsets run
2-36 s, median 10 s, and that is essentially all of the 242 s `ad_shown`.

There is one cause behind almost all of it, and it also explains the USA-bug
false positives below. **A full break does not start with a commercial.** It
starts with 5-20 s of the network's own material — a "USA SPORTS / NASCAR CUP
SERIES" wipe, a produced crowd-and-racing tease, a sponsor billboard over a
clean car shot. That material is made by the same people, scored by the same
music library, and carries the same corner bug as live coverage, so:

- the audio sensor reads it as content — at the 1:43:56 break `p_audio` sits at
  0.01-0.25 for the first 20 s and only then climbs;
- the USA bug check fires on it and votes `content` outright.

Both sensors are being asked to distinguish a network promo from a network
broadcast, which is a genuinely hard problem and arguably the wrong question.
The break's *first frame* is a hard cut out of live coverage, which the picture
shows plainly; the thing that follows it just happens to look and sound like the
show. A dedicated going-to-break bumper detector would attack this directly and
is the obvious next experiment, since those wipes are a small fixed set of
graphics per profile.

---

## Transcription: the operator's idea

Whisper large-v3 over the whole broadcast, on a track reconstructed from the
saved clips — take the last `t[i]-t[i-1]` seconds of each and they tile the
timeline exactly once. 2845 segments, coherent continuous English, 159.6 of
159.6 minutes.

Both halves of the hypothesis turned out to be true. They are not equally
useful, and the *less* promising-sounding one matters more.

### Pre-break cues: real, precise, and too rare to matter much

Cue phrases fired before **5 of 17 breaks**, against **1 of 75 control windows**
(and 0 of the 33 controls drawn from inside breaks). The three clearest:

- 0:01:10 — *"It's all coming up **next** here on USA."* at **−14 s**
- 0:07:07 — *"Green flag from Iowa is **next here on USA**."* at **−8 s**
- 0:36:22 — *"We'll look into that and tell you **when we come back**."* at **−5 s**

The other twelve breaks have no verbal warning at all: commentary continues and
the spot cuts in mid-sentence. Note that **"we'll be right back" never occurs
once** — the phrase everyone expects is not the phrase this crew uses.

So it is a genuine leading indicator with ~29% recall and near-perfect
precision, which is the right shape (it can only pull a switch earlier, never
make one wrong). Wired in as an arming signal that lowers the switch threshold
for 30 s (`cue_policy.py`), across the whole broadcast it is worth:

| | acc% | ad_shown | mean onset lag |
|---|---|---|---|
| state machine | 97.03 | 242 s | 7.5 s |
| + verbal cue | 97.11 | 234 s | 7.1 s |

**Eight seconds over two and a half hours.** It helps three breaks by 2 s each
and does nothing to the three worst onsets (36 s, 12 s, 12 s), because those had
no cue. Worth having if a transcript is on hand for other reasons; not worth
standing up ASR for. Recommendation 7 — a bumper detector — attacks the same
latency far more directly.

### Ad reads over the rejoin: much more common, and a real constraint

This is the half worth acting on. **11 of 17 rejoins carry a sponsor read in the
first 20 seconds of speech** — the picture is back on racing while the audio is
still selling something:

- +0.2 s — *"Brace yourself for something fuel nominal. / Unleaded 88 is cheaper,
  cleaner and greener than regular unleaded"* — a pure spot, over live pictures
- +3.6 s — *"Pump unleaded 88, grown by Iowa corn farmers. / Ally, banking built
  for life today."*
- +3.1 s — *"Corn 350 powered by ethanol here on USA Sports, and this
  never-ending quest for race leader Ryan Blaney…"* — a billboard read blending
  into live commentary mid-sentence, with no boundary between them at all

Two-thirds of rejoins, so this is the norm rather than an edge case, and it is
a **limit on audio classification rather than an opportunity**. It shows up in
the measurement: the audio sensor's worst rejoin lag is 30 s against 20 s for
the visual furniture signal.

**The two sensors should not be weighted equally in both directions** — vision
is the better authority at a rejoin, audio at an onset. That asymmetry is now
measured twice, from opposite directions: in-house *bumpers* fool both sensors
going into a break, and in-house *ad reads* fool audio specifically coming out
of one.

---

## Defects found in the current classifier

**1. The USA bug fires during breaks, hard.** The anchor called `content` on 27
frames whose truth is `ad`, at match scores of **0.91-0.99** — so no threshold
change can fix it. They cluster at 11 distinct break edges, and they are NBC's
own going-to-break and coming-back bumpers, which legitimately carry the USA
wordmark. The bug means "this is USA", not "this is live coverage".

`AGENTS.md` records this check as "0/3000 false positives". That was measured
against archive frames, which under-sample bumpers because bumpers are brief.
Over a continuous broadcast the check is wrong 0.8% of the time it fires, always
in the direction that costs a missed break.

Run-length filtering separates them imperfectly (requiring 6 consecutive frames
removes 14 of 27 but costs 12 s of latency at every rejoin) and is not worth it.
The right fix is the one the state machine already applies: **treat the anchor as
strong evidence, not as an immediate commit.** `stateless` and `debounce2` both
override on it; that is why they flap.

**2. Nothing fires on pylon mode.** Ten minutes of live racing with no OpenCV
verdict — see above.

**3. The debounce default trap is still open.** Re-checked against the current
code, not carried over: `config.py:11` defaults `enable_debounce` to `False`,
`state.py:33` defaults it to `True`, and `main.py:50` copies config over state
unconditionally at startup. Docker is fine because `example.env:29` sets
`RECEIVER_ENABLE_DEBOUNCE=1`, but a dev server started without it silently runs
the `stateless` policy — which flaps **35 times** on this broadcast. Making the
two defaults agree is a one-line fix.

---

## Does any of this transfer?

The signal that ought to transfer is the furniture detector, because it has no
templates: it measures how much of the frame is edge that has not moved for
several seconds. Live sports is pinned under graphics; a national spot is not.

Tested against `server/frames/`, whose manual labels are almost entirely
**NASCAR on Fox** from March 2026 — a different network and graphics package,
with nothing fitted to it:

| | AUC | acc |
|---|---|---|
| Fox archive, blocked by capture date, 169 labelled frames | 0.845 | 0.793 |
| USA/NBC continuous capture | 0.959 | 0.927 |

It transfers, and it weakens. Some of that gap is real and some is the dataset:
the burst archive is enriched in transitions and hard frames by construction
(the previous notes measured 4x), and its 4 s burst spacing makes the window
cover 12-16 s instead of 8 s. The two numbers are **not directly comparable**,
and making them comparable needs a continuous recording of a non-NBC broadcast,
which does not exist yet.

### …but not on the frames that actually break it

`frames/incorrect_labels.json` holds 364 operator "report wrong" entries, 283 of
them genuine per-frame errors (173 missed ads, 110 false ads), all from Fox in
March 2026. This is the classifier's own failure set — the most adversarial test
available. Of those, 121 have a reconstructable burst window.

**The furniture signal is at chance on them.**

| | value |
|---|---|
| fused AUC, blocked by capture date | 0.536 |
| accuracy | 0.587 |
| majority-class baseline | 0.587 |

No single region beats 0.53. With n=121 the standard error on that AUC is about
0.055, so this is not a weak effect — it is indistinguishable from nothing.

That is the most important caveat in this document, and it cuts against the
0.845 above. The furniture detector separates *typical* Fox frames but carries
no information about the ones the pipeline actually gets wrong. Those are
sponsored squeezebacks, in-broadcast promos and produced montages — material
that has broadcast graphics pinned to it and is an ad anyway, or lacks them and
is content anyway. Persistence cannot see that distinction.

So: **furniture is a cheap win on easy frames and no help on hard ones.**

### Audio does transfer to another network

Audio could not be tested against the manual labels — only 4 of the 462 have
audio, and the March labels predate audio capture. But the archive holds 5815
clips from a **Fox/FS1** broadcast on 2026-05-17 where the shipped OpenCV checks
fired, and those checks are independent of audio, so they serve as proxy labels:
`network_logo` → content, `side_by_side` → ad.

Training on USA/NBC and testing there:

| | AUC |
|---|---|
| trained on USA/NBC, tested on FS1 (1523 clips) | **0.915** |
| trained *within* FS1 (first half → second), same architecture | 0.922 |

**The transferred model very nearly matches one trained on FS1 itself.** That is
the strongest evidence in this document that the audio signal is about
commercials in general — loudness mastering, bass content, the absence of a
broadband engine roar — rather than about one production's music beds.

Two caveats that matter operationally:

- **The ranking transfers; the threshold does not.** At the untuned 0.5
  threshold accuracy is 0.852, *below* the 0.887 majority-class baseline, purely
  because this test set is 89% content and the class prior differs. A shipped
  audio sensor needs its operating point set per profile — or better, fed to the
  policy as a score rather than a verdict, which is what the state machine
  already expects.
- Anchored frames are an easy subset by construction: they are the frames where a
  logo was clearly visible. This says nothing about hard frames, which is exactly
  where the furniture signal turned out to be worthless. Only the instantaneous
  4 s model could be tested at all, because the archive is bursts rather than a
  timeline.

---

## Recommendations

1. **Add an audio sensor to the classifier.** Biggest available win, no LLM
   call, 6.8 ms of DSP per clip, right on the one region the OpenCV checks
   cannot see at all, and the only signal here shown to survive a change of
   network. Ship it as a **continuous score feeding the policy, not as another
   hard verdict** — the ranking transfers between broadcasts but the 0.5
   threshold does not.
2. **Replace debounce with a minimum content dwell plus an evidence
   accumulator.** Two ingredients, ~30 lines. Do *not* build the full duration
   model — measured, the ad-side floor and the NON STOP timer contribute
   nothing.
3. **Stop treating the USA bug as an immediate content commit.** It is right
   99.2% of the time it fires, and the 0.8% is systematic and concentrated at
   break edges.
4. **Do not chase the 120 s NON STOP constant.** It is the most striking thing
   in the data and operationally worthless, because the banner check already
   covers the whole break.
5. **Record a continuous broadcast on Fox or HBO Max.** The audio transfer test
   above had to lean on proxy labels and an easy subset; a continuous non-NBC
   recording would settle it properly, and would also be the only way to check
   whether the 120 s NON STOP constant and the 118 s break floor are NBC
   conventions or industry ones. Cheapest experiment left by a wide margin.
6. **Weight vision and audio asymmetrically by direction** — audio leads at
   break onsets, vision leads at rejoins, because of the ad-read overlap.
7. **Build a going-to-break bumper detector.** This is where all the remaining
   latency is. Every full break opens with the same small set of network wipes
   ("USA SPORTS", "NASCAR CUP SERIES", the IOWA wheel), and both current sensors
   read that material as content because it *is* network material. A template
   match on a handful of graphics per profile would switch on the break's first
   frame instead of 10 s into it — and would be far more valuable than any
   further work on the temporal policy.
8. **Don't stand up ASR for the pre-break cues alone.** Measured over the full
   broadcast they fire before 5 of 17 breaks and are worth 8 s of `ad_shown` in
   two and a half hours. The idea is sound and the precision is near-perfect —
   it just does not fire often enough to pay for a speech pipeline, and item 7
   attacks the same latency far more directly. If a transcript exists for other
   reasons, wire it in as an arming signal; it costs nothing.
9. **Expect the audio sensor to lag the rejoin, and design for it.** Two-thirds
   of rejoins carry a sponsor read over live pictures. Do not "fix" this by
   making the audio sensor more aggressive — let vision win that direction.

---

## Decisions taken without asking

This ran overnight, so these were called rather than raised. Each is cheap to
reverse and each would change some numbers.

1. **Nothing in `server/` was modified.** All of this lives in
   `server/experiments/structure/`. The recommendations above are not
   implemented — the audio sensor and the policy change are both real code
   changes with operational consequences, and they are yours to approve.
2. **Two ground-truth conventions** (the NASCAR wipe belongs to the break;
   sponsor cards over live footage are content, produced sponsor spots are ads).
   Both are argued at the top. Disagreeing with the second would move about 30
   frames and would slightly worsen the reported rejoin latency.
3. **The Progressive fan-cam at 2:13:29 is labelled content.** Crowd shots
   carrying sponsor cards, at a rejoin. This is the single least comfortable call
   in the segment list; treating it as `ad` would extend that break by ~10 s.
4. **The whole broadcast was transcribed rather than just the boundaries.** The
   original plan was ~45 minutes of targeted windows. Whisper turned out to run
   at 2x realtime, and a full transcript makes the control comparison honest
   instead of anecdotal — which mattered, because the partial transcript
   suggested cues fire before ~43% of breaks and the full one puts it at 29%.
   Cost: ~90 minutes of wall clock, free overnight.
5. **`ProcessPoolExecutor` was swapped for threads** in `extract_audio.py` rather
   than disabling the sandbox, after it failed to create a socketpair. The FFT
   work releases the GIL so the cost is small.
6. **Audio-only was preferred over audio+furniture** for the headline policy,
   because the fusion measured worse. This is a judgement that 35 segment blocks
   cannot support 120 features; with more broadcasts the fusion would probably
   win.

## What this does not establish

One broadcast, one network, one production. Seventeen breaks is enough to see
that NON STOP is 120 s and not enough to know the tail of the full-break
distribution.

**The honest caveats on the headline numbers:**

- The dwell floors were chosen from the broadcast they are scored on. Blocked CV
  protects the audio model but not the policy parameters. The plateau in the
  sensitivity sweep is the reason to think they are not badly overfitted, not
  proof of it.
- I also chose the feature block (adding instantaneous and delta features to the
  trailing window) after seeing it reduce onset latency on this broadcast. That
  is one round of manual selection against the test set.
- The `audio+furniture` fusion scored *worse* than audio alone (94.87% vs 97.03%
  with the state machine) — 120 features against 35 segment blocks is too many.
  Audio alone is both better and simpler, which is convenient but was not the
  expected result.
- Ground truth for boundary frames is anchored partly on the furniture signal
  that also feeds one of the evidence sets. Every boundary was verified by eye,
  which is what makes it truth, but the two are not fully independent.
- No LLM comparison was possible: there is no local llama.cpp binary and the
  Docker socket is not accessible from this session. Everything above measures
  what is achievable *without* the LLM, which is why the OpenCV-only baseline is
  shown — but the production pipeline's real accuracy, with the LLM in the loop,
  is not measured here and is certainly better than the 81% that row shows.

## Environment problems worth fixing

- **GPU is unavailable to the sandbox.** The NVIDIA driver is loaded (580.173.02)
  but `/dev/nvidia*` is not reachable, so `ctranslate2` reports zero CUDA devices
  and Whisper ran int8 on CPU at ~2x realtime. Transcribing the broadcast took
  ~90 minutes instead of a few. Adding `/dev/nvidia*` to the sandbox device
  allowlist would make ASR and any local model work practical.
- **`ProcessPoolExecutor` cannot start under the sandbox** — it fails with
  `PermissionError: [Errno 1] Operation not permitted` creating its socketpair.
  Worked around with threads (the numpy/scipy work releases the GIL, so the cost
  is small), but it will bite any future parallel script.
- **The Docker socket is not accessible**, so llama.cpp and hdmi-matrix-control
  could not be started to measure anything end to end.
- **`ps` and `pgrep` cannot see other processes** from the sandbox (`/proc` shows
  only a handful of entries), so a long background job can only be monitored by
  watching its output file's mtime. Minor, but it makes "is it still running?"
  surprisingly awkward.

## Code

`server/experiments/structure/`, with `README.md` covering the order things run
in and the traps. `RESULTS.txt` holds the raw output of every analysis in this
note; `run_all.sh` regenerates it from the extracted signals in about a minute.

The three extractors (`extract_visual`, `extract_audio`, `extract_furniture`)
take ~20 minutes together, and `transcribe_all` ~90 minutes, so their outputs are
kept alongside rather than regenerated.
