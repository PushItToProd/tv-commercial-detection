
- [ ] on `/review`, add a field for what network the image is from (should be populated by default for new images)

- [ ] improve `review_ground_truth.py`
  - [x] when I click ad/content/other or edit the note, it should focus the card.
  - [x] add a way to bulk select and confirm or modify classifications. let me select ranges of cards to edit at once in the contact sheet and in the card view by shift+clicking on the first and last image I want to select and let me make edits on all of them. maybe pop up a modal with all the selected images as a filmstrip. 
  - [ ] if I ctrl+shift+click on any image, select the entire contiguous range of images with the same classification as the selected image so I can quickly confirm the accuracy of a whole stretch. make sure this doesn't risk inadvertently overwriting any classifications i've already made -- if there's a stretch of say 50 images the model labeled as content but I marked the first 5 and the last 10 as ads, only select the 6th through 39th images of that stretch. 
  - [x] make sure the currently focused image is visibly highlighted in the contact strip view. put like a nice thick box shadow around it like you do with the cards. then, when switching between card and contact strip view, make the focus box in the new view blink a few times to make it more visible. stop the blinking early if I do anything that would've changed the focus so it won't keep blinking while I'm trying to do something
  - [x] if I click play on an audio player stop any other player(s) already playing audio
  - [x] when i exit the bulk edit modal, focus on the second image i clicked when picking the range (currently the page scrolls to the last selected image before range selection)
  - [x] give me a way to select/focus an image in the contact strip without navigating to it in card view. if I click on an image that isn't currently focused, just focus it. if i click on an image that is currently focused, go to card view
  - [x] give me a shortcut (I'm thinking `c`) to toggle between contact strip and cards view
  - [ ] add a search box to search image titles and notes
  - [ ] data model - allow breaking up a broadcast into contiguous segments -- the regions I've currently marked by just adding notes to a bunch in bulk
  - [ ] transcribe audio clips and show them in the UI
  - [ ] !! make this a reusable tool and process - support multiple races

- [x] move the images in the save_dir to two subdirectories under that directory: `${save_dir}/images` for full sized images and `${save_dir}/thumbnails` for compressed images

- [/] deduplicate images -- figure out how to clean up without losing information (maybe save as symlinks?)
  - [ ] deduplicate images on save if possible -- at least ones with identical md5 hashes, maybe phashes too

- [/] thorny question: how do I modularize this and make it configurable so this app isn't permanently hardcoded to only work on Fox?
