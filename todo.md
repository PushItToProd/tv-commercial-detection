# To Dos

## Extension

- [ ] auto-save config when you click "start" (currently, it reports an error even though there's placeholder config pre-populated)
- [ ] notify the server when the sender extension is started or stopped
- [ ] ensure we only accept images for classification from one tab/sender at a time -- don't want to accidentally DoS myself if I enable this on multiple tabs
- [ ] don't switch if I'm actively interacting with the video player tab (onmouseover?)
- [ ] if I seek multiple times or catch up to live and end up on a commercial after having been on a break, switch after a short delay
  - although maybe this is just how it works as it is
- [ ] just get rid of multi-endpoint config and assume it's going to be linked to one control server from now on

- [ ] +if the classifier service returned its result to the extension, the extension could do things like mute and (if possible) skip ahead automatically until it's not on an ad break anymore
- [ ] package the extension so I can install it permanently in firefox

- [ ] capture the race name and network name and send with the request

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

## Web app

- [ ] take a path to a single folder to use for all outputs and data saved by the server
- [ ] persist state in a better way (SQLite? Redis?) than just keeping it in a dataclass in memory
- [ ] document expected values for `output_settings` in `AppConfig` (`config.py`)
- [ ] use `pydantic-settings` for settings -- https://docs.pydantic.dev/latest/concepts/pydantic_settings/

- [x] deploy everything in docker
  - [x] move the per-app docker-compose config into `~/Code/docker/tv-commercial-detector/docker-compose.yml`
- [x] make everything configurable via env vars
- [x] split flask app into blueprints
- [x] refactor to use Flask's config support https://flask.palletsprojects.com/en/stable/config/
- [-] would websockets be more useful? -> no. SSE instead

### Classification/Receiver

- [ ] I want a better way of tracking, end-to-end, how long it takes to switch after receiving an image. I also want a way to tell if things get backed up
- [ ] handle latency and backpressure -- right now, I have two external components that can have high-ish response times, but I have no way to handle that. 
  - e.g. if I were to use a more intensive prompt for classification that takes >2s to run and then set the browser extension to send screenshost every second, I think I would end up DoSing my server. It would be better if the extension could "feel" that latency and back off
  - I suppose a rudimentary way to do it would be to make all the request processing on the `/receive` endpoint synchronous, so it only sends a response after classification is finished and the switcher has switched (if needed). Of course, the extension's scheduled screenshot sending would need to be tweaked so it could tell if it had gotten a response from the server for its last request yet and skip sending a new screenshot if it hasn't.
- [ ] elegantly handle timeouts from both the llama.cpp server and the HDMI Matrix control server

- [x] support .jpg in addition to .png files so we can compress on the client side
  - [/] review jpg support -- some places still assume png

- [ ] the receiver saves the received image as a file, but then `_classify_image` takes the file path and reads it as base64 -- maybe that can be cut out
- [ ] factor out an enum of classification labels to support more consistent typing
- [ ] keep track of last receive time -- if we haven't gotten a new screenshot in a while (depending on the receive frequency), update the state to reflect possible connection loss and show that in the UI as well

#### Accuracy

- [ ] periodically save some subset of received images along with their responses from the LLM, so later I can review them and find ones that I disagree with
  - [ ] whenever the classification changes for just one iteration (e.g. three consecutive received images get classified as `content`, `ad`, `content` or vice-versa), save all three images and their responses from the LLM
  - [ ] when saving images, associate them with the active prompt

- [ ] improve the data model of how inaccurate frames are saved. dumping them all in a folder with a .json file feels gross
- [ ] increase the number of recent images we retain when reporting an error

#### Prompt

- [ ] right now I have my prompt hardcoded for just Cup on Fox -- I'll need to add separate prompts other series too
  - [ ] add a dropdown to the UI that lets you pick from multiple prompt presets
    - [ ] eventually: detect which series I'm watching using YTTV and/or live feed data (if a race is live)
  - series/networks to handle:
    - [ ] Xfinity on CW
    - [ ] Cup on Amazon Prime
    - [ ] Cup on TNT
    - [ ] Cup on NBC
    - [ ] Trucks

- [x] move prompt into text file
  - [ ] support switching between multiple prompt files

- [x] go through captured screenshots and classify them as "ad" or "content"
  - [ ] for training purposes, record a level of "importance" -- how much do I care about this being classified correctly?

- [ ] maybe include the previous reported state in the prompt to see if that helps -- e.g. `You last reported seeing (an ad|racing).`
  - try including the previous screenshot, too
  - if I hit "Report", include the corrected value in the prompt instead
- [ ] include the broadcast network, racing series, and race name in the prompt
- [ ] having the Fox/FS1 logo in the corner means it's almost always the main broadcast -- how fast would it be to just ask the model if there's a "Fox" logo in the upper right hand corner? would it be faster on average to start by prompting it to check that and then only doing other checks if there isn't one there?
- [ ] I suppose I could also take a set of correctly and incorrectly classified images, feed them to the LLM I'm using to classify them, ask it what it sees, then ask it to generate a prompt for itself with a summary of elements to look for based on the actual classifications.
- [ ] maybe it's fine to block segments with the guys in the booth, too
- [ ] add more categories other than 'ad' and 'race' -- could add 'side-by-side', 'interview', 'booth segment', etc.
  - could try to call out Fox's transitions to and from commercial breaks specifically 
  - possible categories
    - ads
      - `full-screen-ad`
      - `side-by-side-ad`
    - transition
      - `commercial-break-transition`
      - `sponsor-read`
    - content
      - `racing-on-track`
      - `in-car-camera`
      - `reporter-interview`
      - `commentators-talk-to-camera`
      - `reporter-talks-to-camera`
      - `pre-race-ceremonies`
  - simpler categories: `content-0`, `content-25`, `content-50`, `content-75`, `content-100` -- capture a spectrum from 0% content (full-screen ads) to 100% content (racing action on-screen)
  - this could be interesting: `On a scale from 0-100%, rate how much NASCAR racing content this image contains, where 100% is a full-screen image of NASCAR cars racing on track and 0% is nothing to do with NASCAR racing at all. Reply with just the percentage.`
  - maybe even better: `What percentage of this image contains NASCAR racing content? Reply with just the percentage.`
    - -> ask it to grade each image and then react based on the percentages (moving average?) of the last several images. If we go (100, 90, 100, 30), maybe don't switch right away, but if we go (100, 75, 75, 30), then maybe do switch right away.

- [ ] could I just give Claude or some agent access to my `check_classification.py` script and prompt it to iterate on the prompt until we end up with an optimal one?

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

- [ ] to improve the prompt further, grab closed captions/subtitles and include them with the screenshot when sending to the LLM
  - [ ] can we grab subtitles/captions from the `<video>` tag?
- [ ] somehow capture the broadcast audio and use whisper or something with speaker diarization to check if one of the Fox hosts is talking
  - or try using some kind of audio classification model
  - maybe volume and dynamic range would be a viable signal?

### Switching

- [ ] if I manually switch, temporarily pause automatic switching until I enable it again
- [-] if I send two commands to the switcher back-to-back, will it handle them both without me needing to wait for its response?
  - not sure, but I was doing something with switching back and forth, after which my NUC seemingly randomly glitched out -- seems like it's maybe not ideal
  - [ ] try to avoid sending multiple parallel/back-to-back requests to change inputs

- [x] debounce -- require multiple consecutive classifications as ad or racing before switching
- [x] turn off auto-switch when paused
- [x] when I hit on one of the matrix control buttons in the UI, if the request is accepted, the buttons should be temporarily grayed out and an indicator that the request is processing should be shown. when the matrix is done, the UI should be updated again
- [x] move `state.matrix_switching` updates into `apply_matrix_settings()`

### UI

- [ ] seems like the `/is_ad` SSE channel gets disconnected if the server is down for too long (more than a few seconds) -- would be nice to have it auto-reconnect
- [ ] kinda wish I had hot reload on the frontend when I make UI changes

- [ ] right now, the client requests `/is_ad/last_frame?t=${Date.now}` every time it receives a message from the server, even if there's no new image. this should be updated to avoid a pointless fetch if the image hasn't changed

- [ ] move css out into a single style.css file and unify between the /review and /is_ad endpoints
- [ ] maybe use a CSS framework

- [x] show a visual indicator when an ad has been detected and it's about to switch
  - [ ] provide a way to preempt and tell it it's wrong
  - [ ] provide a way to confirm and switch right away

- "Report" button
  - [x] when I click "Report", save the last few images in case I don't manage to hit the button right away
  - [ ] when I click "Report", it should include a unique ID (timestamp?) of the image reported so there's no race condition from hitting it a split second too late
  - [x] when I click "Report", automatically swap back
    - [ ] and maybe temporarily pause auto-switching
    - [ ] ambitious: when I click "Report", temporarily update the classifier prompt to include the relevant screenshot as an example.
      - not sure how long it should be updated for - probably just until the classification changes again
  - [x] show the latest screenshot on the UI so I can tell what I'm marking as wrong
    - [ ] when I tap "Report", show a popup with all the recently captured frames and their classifications. let me pick which ones specifically were classified wrongly and save the whole batch

- [ ] stretch: allow controlling YTTV (pause, rewind, etc.) from the web UI
- [ ] include `incorrect_frames` in the `/review` endpoint so I can classify them
- [ ] if the server stops responding when the UI polls for updates, show that the connection was lost
- [ ] show a counter of the number of seconds since the last image was received
- [ ] the "Report" button stays highlighted on my iPad after I've tapped it (I had to turn off the transition effect because it made the button flash every second)
  - I guess the button gets focused and then doesn't unfocus -- try unsetting the focus when I tap on the background and/or automatically after a delay

- [x] move UI templates into html files
- [x] add a toggle to turn switching on and off
  - [x] add buttons to manually trigger ad/not-ad mode
- [x] when the classification first changes, even if we don't actually switch the switcher yet, update the UI to show it thinks it's about to change
- [x] change "Wrong!" to "Report"
- [x] add UI toggle to enable/disable debounce
