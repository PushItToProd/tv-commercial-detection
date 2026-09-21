- [x] redirect `/` to `/is_ad`

- [ ] on `/review`, add a field for what network the image is from (should be populated by default for new images)

- [x] move the images in the save_dir to two subdirectories under that directory: `${save_dir}/images` for full sized images and `${save_dir}/thumbnails` for compressed images

- [/] deduplicate images -- figure out how to clean up without losing information (maybe save as symlinks?)
  - [ ] deduplicate images on save if possible -- at least ones with identical md5 hashes, maybe phashes too

- [ ] thorny question: how do I modularize this and make it configurable so I don't have to create a new classifier from scratch for every TV series

- [ ] the audio-based ad detection based on bass triggers false positives on rumbling car engines heard on pit road
  - [ ] identify example audio I can provide as a test case to an agent
    - [ ] or just tell Claude "look for clips the audio classifier regards as ads but that I flagged as content"

- [/] experiment with logprobs
  - [x] ~~test effect of using a grammar~~ -> does nothing
  - [ ] run against the full Cook Out 400 and Iowa Corn 350 broadcasts that I've fully classified (this will take about an hour)
  - [x] test the full model classification prompt
