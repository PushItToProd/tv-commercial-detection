#!/usr/bin/env bash

num_images=0
num_incorrect=0

incorrectly_marked_as_ads=()
incorrectly_marked_as_content=()

for f in frames/*.png; do
    ((num_images++))
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
        ((num_incorrect++))
        if [[ "$actual" == "ad" ]]; then
            incorrectly_marked_as_ads+=("$f")
        else
            incorrectly_marked_as_content+=("$f")
        fi
    fi
done

echo "Processed $num_images images, $num_incorrect incorrect classifications."
echo "Num. incorrectly marked as ads: ${#incorrectly_marked_as_ads[@]}"
echo "  Incorrectly marked as ads: ${incorrectly_marked_as_ads[*]}"
echo "Num. incorrectly marked as content: ${#incorrectly_marked_as_content[@]}"
echo "  Incorrectly marked as content: ${incorrectly_marked_as_content[*]}"