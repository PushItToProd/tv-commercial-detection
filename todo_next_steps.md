- [x] update `/review` to load correctly

- [x] more efficient classification UI
  - [x] bigger UI
  - [x] button to find next unclassified image
- [/] => manually classify a bunch of images for testing my OpenCV-based approach


- [x] update `classify.py` classification functions to return a dataclass instead of a dict


- [x] update `/receive` to periodically save images along with their classification in a .jsonl file so I can review them later
  - [x] save the network and event title
  - [x] include the seek time in the video with each screenshot

```
I no longer use the /save endpoint to capture all images. I want to save images more intelligently when they're received for classification.

* Every minute or so, save all the recent_frames for posterity.
* If a classification would get ignored due to the debounce logic (even if debouncing isn't enabled), save the batch of images.
* I want to keep the classificatoin results for saved images (this should be persisted in a .jsonl file). This should include the image name, the result of classify_image (including the classification, reason, and model response), show or page title, network name, and any other salient information.
```

- [x] logo-based detection feels like it should work really well, and in my tests with `check_classification.py`, it only has an 8% error rate, but it seems to be really bad in practice
  - [x] update `/review` to let me manually record features I want to classify -- the network logo in the upper right, the presence of a vertical or horizontal scoreboard, etc.
  - [x] use OpenCV-based logo and edge detection to get even faster (and maybe more reliable)

- [ ] update the prompt to indicate it's likely an ad unless it has race cars


- [ ] thorny question: how do I modularize this and make it configurable so this app isn't permanently hardcoded to only work on Fox?

- [ ] don't save compressed images into the same directory as uncompressed
