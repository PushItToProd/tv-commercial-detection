# Quick-check confidence (logprobs)

How confident are the LLM passes — the quick check
(`llm_match._report_racing_related`) and the full prompt (`classify_by_prompt`
with `prompt_nbc.txt`) — how does their confidence line up with the operator's
rulings, and does a yes/no grammar change anything?

The findings come from three runs. Two small samples from the Cook Out 400 (160
frames) come first. The 500-frame survey over the Cook Out 400 and the Iowa Corn
350 comes last, and where the two disagree the survey's numbers stand.

Survey summary: in production the LLM sees 3% of frames (380 of 11,682), and
78% of those are content. On them the full prompt passes 3 of 82 ads and calls
26 of 298 content frames `ad`. Half of those 26 are frames the operator rated
cheap to get wrong. The quick check changed no verdict and added 133 ms per
frame on average, so it can be removed from `nascar_on_nbc`. Attaching the
clip to the full prompt made no difference. Over the whole broadcast, the audio
sensor calls 101 Cook Out content frames `ad`, four times the LLM's errors on
that broadcast.

Small-sample summary: on ordinary commercials the quick check is confident and correct. On
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
| `results_undecided.jsonl` | 30 ad, 30 content | Random frames the `nascar_on_nbc` OpenCV checks leave undecided. Production also consults the audio sensor before the LLM, which these samples ignore; the survey accounts for it |

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
B=/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com
for b in USA_4K_Cook_Out_400 USA_4K_Iowa_Corn_350; do
  uv run python scripts/fit_audio_model.py check --dir $B/$b \
      --model src/tv_commercial_detector/classifiers/audio_models/nascar_on_nbc.json \
      --dump experiments/quick_check_logprobs/cascade_$b.jsonl
done
uv run python experiments/quick_check_logprobs/survey.py                    # survey_sample.jsonl, survey_raw.jsonl
uv run python experiments/quick_check_logprobs/analyze_survey.py
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
mix, which is weighted toward ads, so here it cost time. The survey measures
the real share: 22% of frames sent to the LLM are ads.

## 500-frame survey (Cook Out 400 and Iowa Corn 350)

`survey.py` asks all four conditions (quick check and full prompt, each
image-only and with the clip) on a 500-frame test set, and
`analyze_survey.py` summarizes it. The test set is meant to stay fixed while
the prompt is iterated on.

### Test set

Where each frame falls in the `nascar_on_nbc` cascade comes from replaying both
recordings through the OpenCV checks and the audio sensor
(`scripts/fit_audio_model.py check --dump`, saved as
`cascade_<broadcast>.jsonl`). Non-excluded frames only:

| | Cook Out 400 | Iowa Corn 350 |
|---|---|---|
| frames | 5881 | 5801 |
| OpenCV calls `ad` (wrong) | 701 (0) | 409 (0) |
| OpenCV calls `content` (wrong) | 4299 (11) | 3431 (45) |
| audio sensor calls `ad` (wrong) | 608 (101, 16.6%) | 805 (10, 1.2%) |
| audio sensor calls `content` (wrong) | 144 (8) | 905 (2) |
| sent to the LLM | 129 (27 ad, 102 content) | 251 (55 ad, 196 content) |

The audio model was fit on Iowa, so its Iowa numbers are in-sample: it decides
more Iowa frames, and more of them correctly, than it would on a held-out
broadcast. The Cook Out numbers are the honest ones.

`survey_sample.jsonl` holds:

- **census**: all 380 frames sent to the LLM, weight 1. Rates over these are
  production rates, with no sampling error from the selection.
- **audio-decided**: 120 of the 2462 frames the audio sensor decides, 30 ad
  and 30 content per broadcast. In each cell, half are plain frames and half
  are spread evenly over the operator's notes and flags. Each carries `weight`,
  the number of frames it stands for. These frames reach the LLM whenever the
  audio sensor abstains (dead capture, a coarse capture interval, a new model).

No clip in either recording was silent, so every frame ran all four
conditions. The run took 8 minutes (76 s, 95 s, 137 s and 161 s per
condition); `survey_raw.jsonl` holds one line per condition and frame, and a
re-run skips what's already there.

### Timing

Medians (p90), prompt caching on:

| Condition | Wall | Prefill | Generation |
|---|---|---|---|
| quick check, image-only (`max_tokens=1`) | 130 ms (167) | 100 ms | 1 token |
| quick check, with audio (`max_tokens=1`) | 170 ms (175) | 124 ms | 1 token |
| full prompt, image-only | 248 ms (296) | 87 ms | 104 ms, 20 tokens |
| full prompt, with audio | 297 ms (339) | 127 ms | 98 ms, 20 tokens |

### Accuracy on the census

Production conditions are the audio variants.

| Condition | Ads passed as content | Content called ad | AUC |
|---|---|---|---|
| quick check, image-only | 24/82 | 5/298 | 0.869 |
| quick check, with audio | 34/82 | 0/298 | 0.957 |
| full prompt, image-only | 2/82 | 24/298 | 0.980 |
| full prompt, with audio | 3/82 | 26/298 | 0.974 |
| quick check then full prompt (production) | 3/82 | 26/298 | |

Split by broadcast, the production cascade passes 1/27 ads and calls 11/102
content frames `ad` on Cook Out, and 2/55 and 15/196 on Iowa.

Three full-prompt replies (all Iowa content, with audio) described the frame
and stopped without a verdict, which production parses as `unknown`. They are
counted as not `ad` above.

The clip made no difference to the full prompt here, in line with the Iowa
measurement in `AGENTS.md`. The 160-frame Cook Out sample suggested it helped;
that sample was built around hard cases and the survey doesn't bear it out.

### What the errors cost

The operator's `care_away` (an ad left on screen) and `care_back` (the race
switched away from) rate each frame's error cost from 0 to 3. The 26 content
frames the production full prompt calls `ad`:

| `care_back` | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| frames | 1 | 13 | 2 | 10 |

The 3 ads it passes have `care_away` 2, 3 and 3.

Grouped by what they show, the census errors are:

- **Pre-race and feature content, 12 frames** (4 rated 3; the five pre-race
  hype frames are rated 1). Examples: in-car and
  driver-with-fans shots in the pre-race hype segment, a track diagram,
  aerials of the track and parking lots, the Diffey segment. The replies
  mostly give the clean-frame rule as the reason: "no scoring strip, no
  network bug".
- **Live racing with no on-screen furniture, 4 frames** (3 rated 3). "Race
  cars on track, no scoring strip, no "LIVE" badge, no network bug." The same
  rule.
- **The Cheddar's squeezeback on Iowa, 7 frames**, ruled content with
  `care_back` 1 ("low-importance post-ad-break chatter"). The left panel
  shows food close-ups beside the race. The prompt says a second video panel
  of commercial content is a side-by-side break, so the model is applying the
  prompt as written. At `care_back` 1 a late return costs little, so this is
  deferred.
- **Everything else, 3 content frames** (all rated 3): a crowd shot, racing
  under a "NASCAR Americana" promo overlay, and racing under a post-race promo
  overlay.
- **Ads passed as content, 3 frames**: a post-break interview at night, a
  NASCAR interstitial card with the announcer calling the race, and an in-car
  view with a DraftKings overlay during a post-break ad read.

Over all 500 frames, the operator groups with the most errors are the pre-race
hype segment (8/13), the sponsor squeezeback on Iowa (8/8, all audio-decided
frames), the post-break chatter (7/7) and the Diffey segment (4/7).

### The quick check adds nothing

On the census, every frame the quick check rejected, the full prompt also
called `ad`. It never corrected the full prompt, at any threshold tried:

| Quick check rejects at | Ads passed as content | Content called ad |
|---|---|---|
| P(yes) ≤ 0.1 | 3/82 | 26/298 |
| P(yes) ≤ 0.5 (production) | 3/82 | 26/298 |
| P(yes) ≤ 0.9 | 3/82 | 29/298 |

It rejected 13% of census frames against a break-even of 57% (170 ms against
297 ms). The mean LLM time per frame is 430 ms with it and 297 ms without.

On the audio-decided sample it rejected 25%, still well short of break-even,
and a 0.9 threshold rescued one ad there.

### Confidence

The full prompt's P(ad) fell between 0.05 and 0.95 on 18 of 380 census
frames. Of its 29 census errors (26 content, 3 ads), 25 were confident. The
small-sample finding holds: the description commits the model before the
verdict token, so P(ad) says little about how hard a frame is.

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
- **The audio sensor's false `ad` calls on the Cook Out 400** (101 of 608,
  16.6%) outnumber the LLM's errors on that broadcast four to one. This is the
  known weakness in `AGENTS.md`: in-car audio and caution laps without pack
  roar.
- **Server sampling defaults**: `/props` reports `top_k=40`, `min_p=0.05`.
  Neither changes the logprobs measured here, but `min_p` removes the minority
  answer from production sampling once it falls below 5% of the majority's
  probability.

## Suggestions

1. **Remove the quick check from `nascar_on_nbc`.** It changed no verdict on
   the census and costs 133 ms per LLM frame on average. Keep
   `_report_racing_related` for the other profiles until they're measured.
2. **Iterate the full prompt against the survey test set.** `survey.py` asks
   all four conditions; for prompt work only `full_audio` (production) and
   perhaps `full_image` are needed, about 2.5 minutes each over 500 frames. A
   `--conditions` and `--prompt` option would let a prompt variant write to its
   own raw file. Targets, in order of cost:
   - the clean-frame rule, which accounts for 16 of the 26 content errors on
     the census, 7 of them rated 3;
   - replies with no verdict (3 of 500), which a stricter closing instruction
     or a grammar on the final line would remove.
3. **Look at the audio sensor on held-out broadcasts.** On the Cook Out 400 it
   produces more wrong switches than the LLM. Raising `ad_threshold` sends more
   frames to the LLM; the survey's audio-decided sample estimates how the full
   prompt does on them (5% of ads passed, 7% of content called `ad`, weighted).
4. **Record P(ad) in `ClassificationResult.signals`** anyway. It rarely
   carries information today, but a prompt that asks for the verdict before the
   description, or samples several descriptions, might change that.
5. **Set temperature 0** on the full prompt, so a frame always gets the same
   verdict.
6. **Run against both full broadcasts** when coverage needs measuring beyond
   the census. For the LLM only the census matters, so this is mostly a check on
   the audio-decided estimates.
