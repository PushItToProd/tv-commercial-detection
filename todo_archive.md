# To Dos

Checkboxes key:

- [ ] to do
- [/] doing
- [-] canceled/abandoned
- [x] done

## Extension

- [x] capture the race name and network name and send with the request

  - Get video title from YouTube TV:

    ```js
    doc.querySelector(".ypc-video-title-text").textContent.trim()
    ```

  - Get network from YouTube TV:

    ```js
    doc.querySelector(".ypc-network-logo").textContent.trim()
    ```

- [x] only capture screenshots if video is playing
- [x] the "stop" button doesn't actually work - it keeps capturing screenshots even after closing the original tab
  - [x] when you click "start", it changes to "stop", but clicking "stop" doesn't do anything
  - [x] when you close and reopen the popup, the button still says "start" and clicking it repeatedly doesn't do anything but trigger another capture
- [x] stop capturing when the tab is closed
- [x] send a signal when the video is paused
- [x] compress/resize images at capture time so we don't have to do it server side
- [x] avoid switching if I'm seeking the video player
- [x] notify the server as soon as I play/pause/seek/etc.
- [x] use the same logic for identifying the `<video>` tag in `track_interactions.js` as in `get_video_bounds.js`
- [-] use the `rect` property in the `ImageDetails` param of `browser.tabs.captureTab()` instead of `cropImage` to crop the source image
  - -> `rect` only works on Firefox - may as well keep it simple
- [-] potentially resize using canvas https://stackoverflow.com/a/39637827
- [-] allow configuring separate intervals for each endpoint
- [x] weird 'No video found' error sometimes
- [-] maybe just get rid of multi-endpoint config and assume it's going to be linked to one control server from now on (but that would prevent `record_broadcast.py` from being useful)
- [x] notify the server when the sender extension is started or stopped (and show this on the UI)

## Web app
- [x] deploy everything in docker
  - [x] move the per-app docker-compose config into `~/Code/docker/tv-commercial-detector/docker-compose.yml`
- [x] make everything configurable via env vars
- [x] split flask app into blueprints
- [x] refactor to use Flask's config support https://flask.palletsprojects.com/en/stable/config/
- [-] would websockets be more useful? -> no. SSE instead
- [x] general project cleanliness
  - [x] add linting
  - [x] add automated tests
    - [x] work out how to test
  - [x] clean up project structure
    - [x] move all code into a `src/` directory
- [x] take a path to a single folder to use for all outputs and data saved by the server
- [x] redirect `/` to `/is_ad`

### Classification/Receiver

- [x] support .jpg in addition to .png files so we can compress on the client side

- [x] enable multiple classification profiles
  - [x] only specify Fox-specific logos in `nascar_on_fox.py`

- [x] multimodal classification with audio capture -- use Qwen3-Omni
  - [x] first just find a way to record a race broadcast with video and audio
- [x] don't save compressed images into the same directory as their originals
- [x] move the images in the save_dir to two subdirectories under that directory: `${save_dir}/images` for full sized images and `${save_dir}/thumbnails` for compressed images
- [x] review jpg support one last time -- some places still assume png
- [x] keep track of last receive time -- if we haven't gotten a new screenshot in a while (depending on the receive frequency), update the state to reflect possible connection loss and show that in the UI as well
- [x] capture an entire race broadcast (or multiple broadcasts) as frames+audio, then have Claude just iterate on ways to consistently and quickly detect ads -- let it churn overnight or w/e and see what it comes up with
- [x] somehow capture the broadcast audio

#### Review

- [x] `/review` can't handle the amount of image on the page (probably overwhelming the dev server) -- paginate
- [x] update `/review` to let me categorize images based on additional features
- [x] record the broadcast name, network, page URL, and seek time with each image
- [x] make `/review` paginated and filterable
- [x] support better filtering of images based on classifications and things (like `view_classification_results.py`)

#### Accuracy

- [x] periodically save some subset of received images along with their responses from the LLM, so later I can review them and find ones that I disagree with
  - [x] whenever the classification changes for just one iteration (e.g. three consecutive received images get classified as `content`, `ad`, `content` or vice-versa), save all three images and their responses from the LLM

#### OpenCV

- [x] manually classify a bunch of images for testing my OpenCV-based approach

##### `logo_match.py`

- [x] `LOGO_PATH` is currently hardcoded -- it should be configurable

#### Prompt

- [x] move prompt into text file
- [x] go through captured screenshots and classify them as "ad" or "content"
- [x] prompt the model to specifically look for certain attributes like scoreboard position and emit it all in JSON
- [x] try using Qwen 2B instead of 4B? - much less accurate than 4B without examples; adding examples makes it take >2s
- [x] try using Qwen 0.8B with my improved prompt
  - -> still not very good
- [x] write up a prompt to send requests to a llama.cpp server to classify screenshots as content or ads
  - [x] but first, just test if it works with no pre-prompting. just send Qwen an image and ask "ad or content?"
- [x] add something to `/is_ad` to let me flag a classification as wrong and save the image for later
- [x] try using multi-shot prompting to improve accuracy
  - -> also made it slower
- [x] try prompting Qwen with multiple samples to improve accuracy
  - -> get slow with too many examples
- [x] record metrics about how long classification takes
  - https://prometheus.github.io/client_python/exporting/http/flask/
- [x] prompt the model to include a confidence score -- not sure it'll help but could be useful in the future
  - ~~maybe if the confidence is high enough, switch without waiting for a second result~~ -- turns out the confidence is always too high. stupid overconfident LLMs...
- [x] maybe include the previous reported state in the prompt to see if that helps -- e.g. `You last reported seeing (an ad|racing).`
  - try including the previous screenshot, too
  - if I hit "Report", include the corrected value in the prompt instead
  - -> this approach didn't work per the hystersis experiments
- [-] ambitious: when I click "Report", temporarily update the classifier prompt to include the relevant screenshot as an example.
  - feels not super useful with the current approach
  - also now I have phashing

#### Audio

- [x] maybe volume and dynamic range would be a viable signal?

#### Future ideas

- [-] ~~to improve the prompt further, grab closed captions/subtitles and include them with the screenshot when sending to the LLM~~
  - [x] can we grab subtitles/captions from the `<video>` tag? -> looks like no

### Switching

- [x] debounce -- require multiple consecutive classifications as ad or racing before switching
- [x] turn off auto-switch when paused
- [x] when I hit on one of the matrix control buttons in the UI, if the request is accepted, the buttons should be temporarily grayed out and an indicator that the request is processing should be shown. when the matrix is done, the UI should be updated again
- [x] move `state.matrix_switching` updates into `apply_matrix_settings()`
- [x] if I manually switch, temporarily pause automatic switching until I enable it again
- [x] if I send two commands to the switcher back-to-back, will it handle them both without me needing to wait for its response?
  - -> yes
- [x] !!! don't debounce if the classification reason is from OpenCV -- switch instantly

### UI
- [x] show a visual indicator when an ad has been detected and it's about to switch
- [x] add second "Report" button to report without switching
- [x] when I click "Report", save the last few images in case I don't manage to hit the button right away
- [x] when I click "Report", automatically swap back
- [x] when I click "Report temporarily pause auto-switching
- [x] show the latest screenshot on the UI so I can tell what I'm marking as wrong when I click "Report
- [x] move UI templates into html files
- [x] add a toggle to turn switching on and off
  - [x] add buttons to manually trigger ad/not-ad mode
- [x] when the classification first changes, even if we don't actually switch the switcher yet, update the UI to show it thinks it's about to change
- [x] change "Wrong!" to "Report"
- [x] add UI toggle to enable/disable debounce
- [x] add a button to just save the last few screenshots
- [x] maybe use a CSS framework
- [x] mobile-friendly UI so I can use it on my phone (another argument for using a CSS framework -- something like Bootstrap would probably make this easier)
- [x] when I tap "Report", show a popup with all the recently captured frames and their classifications. let me pick which ones specifically were classified wrongly and save the whole batch
- [x] display the reason for the categorization on the UI
- [-] ~~stretch: allow controlling YTTV (pause, rewind, etc.) from the web UI~~ -> now a goal of my living-room-control project

## Additional tools

### `annotate_broadcasts.py`

- [x] improve `review_ground_truth.py`
  - [x] when I click ad/content/other or edit the note, it should focus the card.
  - [x] add a way to bulk select and confirm or modify classifications. let me select ranges of cards to edit at once in the contact sheet and in the card view by shift+clicking on the first and last image I want to select and let me make edits on all of them. maybe pop up a modal with all the selected images as a filmstrip.
  - [x] make sure the currently focused image is visibly highlighted in the contact strip view. put like a nice thick box shadow around it like you do with the cards. then, when switching between card and contact strip view, make the focus box in the new view blink a few times to make it more visible. stop the blinking early if I do anything that would've changed the focus so it won't keep blinking while I'm trying to do something
  - [x] if I click play on an audio player stop any other player(s) already playing audio
  - [x] when i exit the bulk edit modal, focus on the second image i clicked when picking the range (currently the page scrolls to the last selected image before range selection)
  - [x] give me a way to select/focus an image in the contact strip without navigating to it in card view. if I click on an image that isn't currently focused, just focus it. if i click on an image that is currently focused, go to card view
  - [x] give me a shortcut (I'm thinking `c`) to toggle between contact strip and cards view
  - [x] !! make this a reusable tool and process - support multiple races


### `check_classification.py`
