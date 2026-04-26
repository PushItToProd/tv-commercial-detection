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

### Classification/Receiver

- [x] support .jpg in addition to .png files so we can compress on the client side

- [x] enable multiple classification profiles
  - [x] only specify Fox-specific logos in `nascar_on_fox.py`

#### Review

- [x] `/review` can't handle the amount of image on the page (probably overwhelming the dev server) -- paginate
- [x] update `/review` to let me categorize images based on additional features
- [x] record the broadcast name, network, page URL, and seek time with each image

#### Accuracy

- [x] periodically save some subset of received images along with their responses from the LLM, so later I can review them and find ones that I disagree with
  - [x] whenever the classification changes for just one iteration (e.g. three consecutive received images get classified as `content`, `ad`, `content` or vice-versa), save all three images and their responses from the LLM

#### OpenCV

- [x] manually classify a bunch of images for testing my OpenCV-based approach

##### `logo_match.py`

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

#### Future ideas

### Switching

- [x] debounce -- require multiple consecutive classifications as ad or racing before switching
- [x] turn off auto-switch when paused
- [x] when I hit on one of the matrix control buttons in the UI, if the request is accepted, the buttons should be temporarily grayed out and an indicator that the request is processing should be shown. when the matrix is done, the UI should be updated again
- [x] move `state.matrix_switching` updates into `apply_matrix_settings()`
- [x] if I manually switch, temporarily pause automatic switching until I enable it again
- [x] if I send two commands to the switcher back-to-back, will it handle them both without me needing to wait for its response?
  - -> yes

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
