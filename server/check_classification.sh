#!/usr/bin/env bash

for f in frames/*.png; do
    echo "Processing $f"
    classification=$(python classify.py "$f")

    if [[ "$classification" == "race" ]]; then
      classification="content"
    fi

    # frames/labels.json is an object with file names as keys and values as the classification results. get the value for the current file
    actual=$(jq -r --arg file "$(basename "$f")" '.[$file]' frames/labels.json)

    if [ "$classification" == "$actual" ]; then
        echo "$f - correct"
    else
        echo "$f - incorrect (classified as $classification, expected $actual)"
    fi
done