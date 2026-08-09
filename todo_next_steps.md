- [/] make `/review` paginated and filterable
  - [ ] use YYYY-MM-DD date formats for date range filter and include the time

- [ ] on `/review`, add a field for what network the image is from
- [x] move the images in the save_dir to two subdirectories under that directory: `${save_dir}/images` for full sized images and `${save_dir}/thumbnails` for compressed images

- [ ] deduplicate images -- figure out how to clean up without losing information (maybe save as symlinks?)
  - [ ] deduplicate images on save if possible -- at least ones with identical md5 hashes, maybe phashes too

- [/] thorny question: how do I modularize this and make it configurable so this app isn't permanently hardcoded to only work on Fox?
