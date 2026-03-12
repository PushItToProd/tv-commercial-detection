# To Dos

- [ ] right now I have things hardcoded for just Cup on Fox -- I'll need to add separate prompts other series too
  - [ ] add a dropdown to the UI that lets you pick from multiple prompt presets
    - [ ] eventually: detect which series I'm watching using YTTV and/or live feed data (if a race is live)
  - series/networks to handle:
    - [ ] Xfinity on CW
    - [ ] Cup on Amazon Prime
    - [ ] Cup on TNT
    - [ ] Cup on NBC
    - [ ] Trucks

- [ ] show a visual indicator when an ad has been detected and it's about to switch
  - [ ] provide a way to preempt and tell it it's wrong
  - [ ] provide a way to confirm and switch right away

- [ ] elegantly handle timeouts from both the llama.cpp server and the HDMI Matrix control server

## Extension

- [x] only capture screenshots if video is playing
- [ ] auto-save config when you click "start"
- [x] the "stop" button doesn't actually work - it keeps capturing screenshots even after closing the original tab
  - [x] when you click "start", it changes to "stop", but clicking "stop" doesn't do anything
  - [x] when you close and reopen the popup, the button still says "start" and clicking it repeatedly doesn't do anything but trigger another capture
- [x] stop capturing when the tab is closed
- [x] send a signal when the video is paused
- [ ] notify the server when the sender stops or starts
- [ ] if I'm actively interacting with the video player, avoid switching
- [ ] allow configuring separate intervals for each endpoint
- [ ] compress/resize images at capture time -- `browser.tabs.captureTab()` takes an `ImageDetails` as its second arg
  - https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/tabs/captureTab
  - https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/API/extensionTypes/ImageDetails
    - `{format: "jpeg", quality: 60}` 
    - [ ] use `rect` instead of `cropImage`
  - potentially resize using canvas https://stackoverflow.com/a/39637827

- [ ] +if the classifier service returned its result to the extension, the extension could mute and (if possible) skip ahead automatically until it's not on an ad break anymore
- [ ] package the extension so I can install it permanently in firefox

## Web app

- [x] deploy everything in docker
  - [ ] move the per-app docker-compose config into `~/Code/docker/tv-commercial-detector/docker-compose.yml`
- [/] make everything configurable via env vars
- [x] split flask app into blueprints
- [ ] refactor to use Flask's config support https://flask.palletsprojects.com/en/stable/config/
- [ ] persist state in a better way than just keeping it in a dataclass in memory
- [ ] move outputs out of `server/` directory into a configurable folder someplace else

### Classification/Receiver

- [ ] keep track of last receive time, if we haven't gotten a new screenshot in a while (depending on the receive frequency), update the state to reflect possible connection loss

- [ ] periodically save some subset of received images along with their responses from the LLM, so later I can review them and find ones that I disagree with
  - [ ] whenever the classification changes for just one iteration (e.g. three consecutive received images get classified as `content`, `ad`, `content` or vice-versa), save all three images and their responses from the LLM

- [x] move prompt into text file
  - [ ] support switching between multiple prompt files

- [x] go through captured screenshots and classify them as "ad" or "content"
  - [ ] for training purposes, record a level of "importance" -- how much do I care about this being classified correctly?
- [x] write up a prompt to send requests to a llama.cpp server to classify screenshots as content or ads
  - [x] but first, just test if it works with no pre-prompting. just send Qwen an image and ask "ad or content?"

- [x] add something to `/is_ad` to let me flag a classification as wrong and save the image for later
- [x] try using multi-shot prompting to improve accuracy
  - -> also made it slower
- [x] try prompting Qwen with multiple samples to improve accuracy
  - -> slower with too many examples
- [x] I suppose I could also take a set of correctly and incorrectly classified images, feed them to the LLM I'm using to classify them, ask it what it sees, then ask it to generate a prompt for itself with a summary of elements to look for based on the actual classifications.

- [ ] record metrics about how long classification takes

- [x] prompt the model to specifically look for certain attributes like scoreboard position and emit it all in JSON
- [ ] maybe include the previous reported state in the prompt to see if that helps -- e.g. `You last reported seeing (an ad|racing).`
  - try including the previous screenshot, too
  - if I hit "Wrong!", include the corrected value in the prompt instead
- [ ] prompt the model to include a confidence score -- not sure it'll help but could be useful in the future
  - [ ] maybe if the confidence is high enough, switch without waiting for a second result

- [ ] include the broadcast network, racing series, and race name in the prompt

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

- [x] try using Qwen 2B instead of 4B? - much less accurate than 4B without examples; adding examples makes it take >2s
- [x] try using Qwen 0.8B with my improved prompt
  - -> still not very good

- [ ] support .jpg in addition to .png files

#### Improvement ideas

- [ ] grab closed captions/subtitles and include them with the screenshot when sending to the LLM
  - [ ] can we grab subtitles/captions from the `<video>` tag?
- [ ] somehow capture the broadcast audio and use whisper or something with speaker diarization to check if one of the Fox hosts is talking

### Switching

- [x] debounce -- require multiple consecutive classifications as ad or racing before switching
- [ ] if I manually switch, temporarily pause automatic switching until I enable it again
- [x] turn off auto-switch when paused
- [ ] if I send two commands to the switcher back-to-back, will it handle them both without me needing to wait for its response?
  - not sure, but I was doing something with switching back and forth, after which my NUC seemingly randomly glitched out -- seems like it's maybe not ideal

### UI

- [ ] move UI templates into html files
- [x] add a toggle to turn switching on and off
  - [x] add buttons to manually trigger ad/not-ad mode
- "Wrong!" button
  - [x] when I click "Wrong!", save the last few images in case I don't manage to hit the button right away
  - [ ] when I click "Wrong!", automatically swap back and maybe temporarily pause auto-switching
    - [ ] ambitious: when I click "Wrong!", temporarily update the classifier prompt to include the relevant screenshot as an example. 
      - not sure how long it should be updated for - probably just until the classification changes again
  - [ ] show the latest screenshot on the UI so I can tell what I'm marking as wrong

- [ ] stretch: allow controlling YTTV (pause, rewind, etc.) from the web UI

- [ ] include `incorrect_frames` in the `/review` endpoint so I can classify them

- [ ] if the server stops responding when the UI polls for updates, show that the connection was lost

- [ ] when the classification first changes, even if we don't actually switch the switcher yet, update the UI to show it thinks it's about to change

- [ ] show a counter of the number of seconds since the last image was received
