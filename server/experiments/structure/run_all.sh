#!/usr/bin/env bash
# Regenerate RESULTS.txt from the extracted signals.
#
# Assumes the three extractors have already run (see README.md) — they take
# ~20 minutes and re-reading 4775 images every time is not worth it. Speech is
# also excluded: transcribe_all.py takes ~90 minutes on CPU and its output is
# checked in.
set -euo pipefail
cd "$(dirname "$0")/../.."

{
  echo "# Raw output of every analysis behind notes/broadcast-structure-2026-08.md"
  echo "# Regenerate with: bash experiments/structure/run_all.sh"
  for s in ground_truth analyse_structure audio_probe evidence evaluate sensitivity \
           cross_check_frames check_corrections cross_check_audio \
           transcript_cues cue_policy; do
    echo
    echo "################ $s.py ################"
    echo
    uv run python "experiments/structure/$s.py" 2>&1
  done
} > experiments/structure/RESULTS.txt
echo "wrote experiments/structure/RESULTS.txt"
