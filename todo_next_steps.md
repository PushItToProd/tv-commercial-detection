- [ ] move the images in the save_dir to two subdirectories under that directory: `${save_dir}/images` for full sized images and `${save_dir}/thumbnails` for compressed images

- [x] enable multiple classification profiles
  - [x] only specify Fox-specific logos in `nascar_on_fox.py`

- [ ] deduplicate images -- figure out how to clean up without losing information (maybe save as symlinks?)
  - [ ] deduplicate images on save if possible -- at least ones with identical md5 hashes, maybe phashes too

- [ ] thorny question: how do I modularize this and make it configurable so this app isn't permanently hardcoded to only work on Fox?

- [ ] don't save compressed images into the same directory as uncompressed
