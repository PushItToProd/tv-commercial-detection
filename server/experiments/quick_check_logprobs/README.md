# Quick-check confidence (logprobs)

How confident are the LLM passes — the quick check
(`llm_match._report_racing_related`) and the full prompt (`classify_by_prompt`
with `prompt_nbc.txt`) — how does their confidence line up with the operator's
rulings on the Cook Out 400, and does a yes/no grammar change anything?

Summary: on ordinary commercials the quick check is confident and correct. On
NASCAR-themed ads the image-only prompt is confidently wrong, because it asks
whether the image contains "anything related to NASCAR", and those ads do. The
audio prompt production runs asks the right question and is rarely confidently
wrong, but it is unsure on most ads, so its yes/no reply throws away signal that
its P(yes) carries. A threshold on P(yes) would reject more ads than the reply
does. A grammar has no effect on speed or on the probabilities.

The full prompt passes almost no ads (0 of 30 ordinary, 2 of 33 hard), but
calls 1 in 6 undecided content frames an ad, nearly always with certainty: its
verdict follows the description it has just written, so the verdict token's
probability is 0 or 1 and carries almost no information. Most of those errors
are pre-race feature content that the prompt's "completely clean frame" rule
calls an ad. On these frames the quick check changed no verdict the full prompt
would have reached on its own.

## Method

Recording: `tv.youtube.com/USA_4K_Cook_Out_400`, rulings from
`annotations.json` (frames marked `exclude` skipped). Model: Qwen3-Omni-30B-A3B
Q4_K_M on llama.cpp b11058 at `gmktec.zane.network:3002`.

Each frame was asked both quick-check prompts at temperature 0 with
`logprobs=True, top_logprobs=20`:

- **image-only**: "Does this image contain anything related to NASCAR racing?
  Reply with only 'yes' or 'no'."
- **with audio**: the prompt production uses when `enable_llm_audio` is set
  (it is, in `config.json`) and the clip isn't silent, with the frame's real
  clip. None of the sampled clips were silent.

P(yes) is the probability on yes-tokens (`yes`, `Yes`, ` yes`, …) divided by
the probability on yes plus no. Tokens other than yes/no never got more than
0.9% of the first token's probability, and the reply always matched the argmax.
"Yes" passes a frame on to the full prompt, and "no" rejects it as an ad.

Two samples:

| File | Frames | Selection |
|---|---|---|
| `results.jsonl` | 50 ad, 50 content | Stratified: 3–5 frames from each operator note (Hamlin, Briscoe, Wallace/Reddick, Americana promo, post-break ad read, Diffey segment, Americana ad read, squeezeback), plus `care_away`&`care_back` both nonzero, `fraught`, bumper, side-by-side, studio, live race, random fill |
| `results_undecided.jsonl` | 30 ad, 30 content | Random frames the `nascar_on_nbc` OpenCV checks leave undecided, which are the frames production sends to the LLM |

The stratified sample was built to contain the hard cases and ended up with
almost no ordinary spots, which is why the second sample exists. Consecutive
frames from one note are highly correlated, so the effective sample size is well
below the frame count. Treat every number here as a direction.

```bash
uv run python experiments/quick_check_logprobs/run.py                       # results.jsonl
uv run python experiments/quick_check_logprobs/run.py --undecided 30 \
    --out experiments/quick_check_logprobs/results_undecided.jsonl
uv run python experiments/quick_check_logprobs/analyze.py [results_undecided.jsonl]
uv run python experiments/quick_check_logprobs/grammar_test.py [--audio]
uv run python experiments/quick_check_logprobs/full_prompt.py               # results_full_prompt.jsonl
uv run python experiments/quick_check_logprobs/analyze_full_prompt.py
```

## Findings

### Ordinary ads: confident and right

On the random undecided sample:

| Prompt | Ads rejected at 0.5 | Content wrongly rejected at 0.5 | AUC |
|---|---|---|---|
| image-only | 27/30 (most at P(yes) < 0.02) | 1/30 | 0.956 |
| with audio | 22/30 | 0/30 | 0.997 |

AUC is the probability that a random content frame scores a higher P(yes) than
a random ad frame.

The three ads the image-only prompt passed were an "xfinity | NASCAR Proud
Premier Partner" card (1.000), a Briscoe spot (0.981) and a Bass Pro Shops
flag shot (0.810).

### NASCAR-themed ads: confidently wrong on the image-only prompt

In the stratified sample, all five Wallace/Reddick frames scored P(yes)
0.956–0.994, and all three bumpers scored 1.000. Other examples: an ad showing
Earnhardt's #3 Goodwrench car scored 0.999, and a newspaper clipping about a
harness race in the Diffey segment (content) scored 0.263.

These are correct answers to the literal question. A Hamlin firesuit ad does
contain something related to NASCAR. The audio prompt asks "is this a NASCAR
broadcast (not an ad)?", which is the question the check needs answered.

On the stratified frames that OpenCV leaves undecided (33 ad, 16 content):

| Prompt | Wrong | Confident (>95% one way) | Leaning (80–95%) | Coin toss (50–80%) | AUC |
|---|---|---|---|---|---|
| image-only | 31/49 | 20 | 5 | 6 | 0.58 |
| with audio | 28/49 | 9 | 12 | 7 | 0.81 |

The audio prompt pulls the hard ads toward the middle; one Hamlin frame moved
from 0.971 to 0.161. It doesn't rescue all of them: three of four Americana
promo frames stayed at 0.98+ and all three bumpers at 0.94+.

### The model's doubt partly matches the operator's

Confidence below is |2·P(yes) − 1|: 0 is a coin toss and 1 is certain.
Stratified sample, image-only prompt:

| Group | Ad mean confidence | Content mean confidence |
|---|---|---|
| `care_away` and `care_back` both nonzero | 0.72 (n=14) | 0.998 (n=10) |
| `fraught` | 0.79 (n=19) | 0.79 (n=12) |
| no note, not fraught, not ambiguous | 0.99 (n=13) | 0.997 (n=28) |

On ads, the frames the operator flagged are the frames the model is less sure
about. On content it isn't: the ambiguous content frames are the "NASCAR
Americana" ad read over racing pictures, and the model is certain because the
picture is racing. The model registers visual ambiguity. The operator's
`care_*` fields encode the cost of a wrong switch in each direction, which the
picture can't show.

The audio prompt is unsure on nearly every ad, ordinary spots included (mean
confidence 0.61 on unflagged ads), so its low confidence doesn't single out the
hard frames.

### P(yes) carries more than the reply

With the audio prompt, content frames almost all score ≥ 0.95, while ordinary
ads spread from 0.02 to 0.88. The reply cuts at 0.5 and passes most of those
ads on.

| Rule (audio prompt) | Ordinary ads rejected (undecided sample) | Content wrongly rejected (undecided sample) | Hard content wrongly rejected (stratified, OpenCV undecided) |
|---|---|---|---|
| reply is "No" (P(yes) < 0.5) | 22/30 | 0/30 | 1/16 |
| P(yes) < 0.9 | 30/30 | 1/30 | 2/16 (both Diffey segment) |

A 0.9 threshold looks better on these samples, but a wrongly rejected content
frame switches away from the race (`care_back` is 3 on most content), and the
samples are too small to settle the tradeoff. The full-broadcast run below is
what should pick a threshold.

### Adding audio costs ordinary-ad recall at 0.5

With the reply as the decision, the audio prompt rejects 22/30 ordinary spots
against the image-only prompt's 27/30. Six spots the image-only prompt
rejects at P(yes) < 0.08 get P(yes) 0.51–0.88 with audio and go on to the full
prompt; one spot moves the other way (0.810 to 0.337). Thresholding P(yes)
recovers them (see above).

### Temperature 0.2 makes middle-band frames non-deterministic

Production samples at temperature 0.2. That sharpens a probability p to
p⁵ / (p⁵ + (1−p)⁵), so a frame at 0.6 answers yes about 88% of the time and
frames between roughly 0.4 and 0.6 flip between runs. With the audio prompt,
13 of the 100 stratified frames and 4 of the 60 undecided frames sat in that
band; with the image-only prompt, 4 and 0.

### A yes/no grammar changes nothing

`grammar_test.py` re-asked the 60 undecided frames three ways, rotating the
order per frame, with `cache_prompt: false` so every request paid the full
image prefill. Timings are from llama.cpp's `timings` block.

| Condition | image-only wall / prefill (median) | with-audio wall / prefill (median) | Tokens generated |
|---|---|---|---|
| baseline (`max_tokens=10`) | 131 / 103 ms | 182 / 148 ms | 2 (answer + end) |
| GBNF `[yY][eE][sS] \| [nN][oO]` | 145 / 102 ms | 182 / 147 ms | 2 |
| `max_tokens=1`, no grammar | 116 / 101 ms | 175 / 147 ms | 1 |

- **Speed**: prefill (encoding the image, audio and prompt) is 80% of each
  request. Generation is two tokens at about 3 ms each. A grammar can't shorten
  either; the model already emits exactly the answer and an end token. The
  14 ms image-only difference didn't reproduce with audio and is likely noise
  or grammar setup cost. Capping `max_tokens` at 1 skips the end token and
  saves 7–15 ms.
- **Probabilities**: identical to four decimal places with and without the
  grammar, and no verdict flipped. llama.cpp reports probabilities before the
  grammar masks tokens, and the model puts at most 0.4% of its probability on
  anything other than yes/no on these frames, so there is nothing for a grammar
  to remove.

A grammar would only help a model that wanders (a leading space, "Yes.", a
sentence). Qwen3-Omni doesn't here. It could still serve as insurance against a
model swap, but `max_tokens=1` plus reading P(yes) gives the same protection
and a small speedup.

### Full prompt

`full_prompt.py` re-asked the 160 frames from both samples with `prompt_nbc.txt`
at temperature 0, image-only and with the clip, and `analyze_full_prompt.py`
summarizes the result (`results_full_prompt.jsonl`). Verdicts were re-read from
`annotations.json`, so the four side-by-side frames since corrected from
`content` to `ad` count as ads here.

The reply is a description of up to 15 words ending in `type=ad` or
`type=racing`. P(ad) is read where the verdict is chosen. Qwen's tokenizer
writes `=r` as one token and `=` `ad` as two, so the choice happens at the
token holding the `=`. Every reply parsed, P(ad) agreed with the parsed verdict
in all 320 replies, and tokens other than the two verdicts got at most 0.02%.

#### Timing

Medians over 160 frames, llama.cpp's default prompt caching on (as in
production). The prompt text precedes the image, so its ~1170 tokens are
reused across requests and only the image, audio and closing tokens are encoded.

| Condition | Wall | Prefill | Generation | Tokens generated |
|---|---|---|---|---|
| image-only | 256 ms (p90 306) | 86 ms | 108 ms | 21 (p90 31) |
| with audio | 306 ms (p90 367) | 127 ms | 107 ms | 22 (p90 31) |

Unlike the quick check, a third of the time goes to generating the
description, at about 5 ms per token. A request that misses the cached text
pays about 100 ms more prefill; the first request of the run took 563 ms.

The production path for a frame OpenCV leaves undecided (quick check with
audio, then full prompt with audio) costs about 470 ms when the quick check
passes the frame on.

#### Accuracy

| Sample | Condition | Ads passed as content | Content called ad | AUC |
|---|---|---|---|---|
| random undecided (30 ad, 30 content) | image-only | 0/30 | 8/30 | 0.988 |
| | with audio | 0/30 | 5/30 | 0.989 |
| stratified, OpenCV undecided (33 ad, 16 content) | image-only | 4/33 | 10/16 | 0.77 |
| | with audio | 2/33 | 5/16 | 0.91 |

The full prompt's errors run the opposite way from the quick check's. It
passes on almost no ads, including the NASCAR-themed spots that fool the quick
check: every Hamlin, Briscoe, Wallace/Reddick and bumper frame was called `ad`
under both conditions. Its errors are content called `ad`. The clip helps under
every breakdown.

What it gets wrong, with audio, on frames that reach the LLM:

- **Pre-race feature content**: cars and crew in a garage, a driver signing
  autographs, a helmet close-up, a fan event. The replies say so in as many
  words — "no live race or scoring strip", "no race track or broadcast
  graphics visible" — which is the prompt's "COMPLETELY clean frame" rule
  firing on feature content.
- **The Diffey "Did You Know" segment**: 4 of 5 frames `ad`, including the
  title card and the newspaper clipping.
- **Ads it passes**: the #3 Goodwrench car ("No network bugs, score strips, or
  banners" — and still `type=racing`), and a post-break Bank of America aerial.

The "NASCAR Americana" ad read and the Credit One squeezeback are also called
`ad` (the squeezeback despite the prompt's rule for it), but OpenCV decides
most of those frames before the LLM sees them.

#### Confidence

With audio, 7 of the 109 frames that reach the LLM got P(ad) between 0.05 and
0.95; image-only, 4. Of the 14 wrong answers with audio in the stratified
sample, 12 were confident. The model writes its description first and the
verdict follows from it, so by the verdict token the decision is already made.
The operator's flags barely register: mean confidence is 0.90–0.98 on
ambiguous and fraught frames against 0.97–1.00 on unflagged ones.

A useful confidence from this pass would need to come from somewhere else:
sampling the description several times at a nonzero temperature and counting
verdicts, or taking the verdict's probability without a description in front
of it.

#### What the quick check adds

On the 109 frames that reach the LLM, running the quick check (audio prompt,
reply as the decision) before the full prompt changed no verdict:

| Operator | Quick check | Full prompt | Frames |
|---|---|---|---|
| ad | no | ad | 28 |
| ad | yes | ad | 33 |
| ad | yes | content | 2 |
| content | no | ad | 1 |
| content | yes | ad | 9 |
| content | yes | content | 36 |

Every frame the quick check rejected, the full prompt would also have called
`ad`. On these frames the quick check only saves time: about 306 ms on each
frame it rejects, at a cost of about 165 ms (the quick check with
`max_tokens=1`) on each frame it passes on. That breaks even when it rejects
more than about 54% of the frames reaching the LLM. It rejected 27% of this
mix, which is weighted toward ads, so here it cost time. The real share of ads
among undecided frames in a broadcast would settle it.

## Side observations

- **OpenCV calls the post-break ad read `content`.** The USA bug stays up over
  the crowd shot with the sponsor inset, so 5 stratified ad frames got OpenCV
  verdict `content`. The image-only quick check scored those frames 0.05–0.16,
  which is correct, but production never asks it because OpenCV decides first.
  This is a known issue.
- **Four frames tagged `video: side_by_side` were ruled `content`** and got
  OpenCV verdict `ad` from the NASCAR NON STOP banner. They were a misclicked
  side-by-side break, since corrected to `ad`. `results.jsonl`, and the
  quick-check numbers above that include OpenCV-decided frames, still count
  them as content. None of them reach the LLM, so the OpenCV-undecided numbers
  are unaffected.
- **Server sampling defaults**: `/props` reports `top_k=40`, `min_p=0.05`.
  Neither changes the logprobs measured here, but `min_p` removes the minority
  answer from production sampling once it falls below 5% of the majority's
  probability.

## Suggestions

1. **Fix the full prompt's clean-frame rule** before tuning anything else. It
   causes most of the content called `ad` on frames that reach the LLM. Pre-race
   features (garage, autographs, driver close-ups, the Diffey segment) are
   produced without a scoring strip or network bug. Candidates: limit the rule
   to frames showing race cars on track, or add pre-race feature material to
   the race-broadcast indicators.
2. **Run a larger sample (~500 frames).** At the timings above, asking both
   quick-check prompts and both full-prompt conditions takes about 0.9 s per
   frame (quick check ~0.3 s, full prompt ~0.56 s), so 500 frames is about
   7–8 minutes; the production conditions alone (quick check and full prompt,
   both with audio) take about 4. Concurrent requests don't help, since the
   server is prefill-bound. `run.py` samples; `full_prompt.py` currently reads
   its frames from `run.py`'s output files.
3. **Decide whether to keep the quick check.** It changed no verdict here and
   only pays for itself if it rejects more than about half the frames reaching
   the LLM. The larger sample (or a full-broadcast run) can measure that share.
   If it stays, pick a P(yes) threshold from the larger sample instead of the
   0.5 the reply implies, broken down by operator note and `care_*`.
4. **Get a usable confidence out of the full prompt**, if one is wanted: count
   verdicts over several sampled descriptions, or ask for the verdict before the
   description (and measure what that costs in accuracy).
5. **Record P(yes) and P(ad) in `ClassificationResult.signals`.** Logprobs cost
   nothing measurable, and recording them in `classifications.jsonl` would
   allow calibrating on live broadcasts.
6. **Try shortening the description.** A third of the full prompt's time goes
   to generating ~22 tokens. Whether the description helps accuracy is untested.
7. **Reword the image-only quick-check prompt** for frames with a silent or
   missing clip, to ask whether the frame is live race coverage or a
   commercial, and compare it against the current wording on the NASCAR-themed
   ads.
8. **Try fusing instead of cascading.** The LLM probabilities and the audio
   sensor's p(ad) are all continuous. A logistic regression on them, fit the
   way `fit_audio_model.py` fits the audio model, might beat running them in
   sequence with a hard cutoff at each step.
9. **Set temperature 0** on both passes, so a frame always gets the same
   verdict. (`max_tokens=1` on the quick check is done: ~9.5% faster.)
10. **Run against both full broadcasts** (Cook Out 400 and Iowa Corn 350, about
    12,000 frames) when coverage needs measuring more comprehensively. Run
    OpenCV first and record its verdict. `run.py` would need a mode that takes
    every frame and resumes from existing output, keyed on filename.
