# To Dos

Checkboxes key:

- [ ] to do
- [/] doing
- [-] canceled/abandoned
- [x] done

## Extension

- [ ] !! it has a terrible memory leak -- Firefox was using _gigs_ of RAM after I had it running for a few hours

- [ ] auto-save config when you click "start" (currently, it reports an error even though there's placeholder config pre-populated)
- [ ] weird 'No video found' error sometimes
- [ ] notify the server when the sender extension is started or stopped
- [ ] ensure we only accept images for classification from one tab/sender at a time -- don't want to accidentally DoS myself if I enable this on multiple tabs
- [ ] don't switch if I'm actively interacting with the video player tab (onmouseover?)
- [ ] if I seek multiple times or catch up to live and end up on a commercial after having been on a break, switch after a short delay
  - although maybe this is just how it works as it is
- [ ] just get rid of multi-endpoint config and assume it's going to be linked to one control server from now on

- [ ] +if the classifier service returned its result to the extension, the extension could do things like mute and (if possible) skip ahead automatically until it's not on an ad break anymore
- [ ] package the extension so I can install it permanently in firefox

## Web app

- [ ] general project cleanliness
  - [ ] add linting
  - [ ] add automated tests
    - [ ] work out how to test
  - [ ] clean up project structure
    - [ ] move all code into a `src/` directory
    - [ ] store outputs (`frames`, outputs from `check_classification.py`, etc.) in a separate folder not intermingled with code

- [ ] take a path to a single folder to use for all outputs and data saved by the server
- [ ] persist state in a better way (SQLite? Redis?) than just keeping it in a dataclass in memory
- [ ] document expected values for `output_settings` in `AppConfig` (`config.py`)
- [ ] use `pydantic-settings` for settings -- https://docs.pydantic.dev/latest/concepts/pydantic_settings/

### Classification/Receiver

- [ ] parallelize processing -- send images to the LLM as fast as it can handle them but, while waiting for the response, keep processing incoming images with OpenCV in case they have a strong signal of an ad or not-ad

- [ ] I want a better way of tracking, end-to-end, how long it takes to switch after receiving an image. I also want a way to tell if things get backed up
- [ ] handle latency and backpressure -- right now, I have two external components that can have high-ish response times, but I have no way to handle that. 
  - e.g. if I were to use a more intensive prompt for classification that takes >2s to run and then set the browser extension to send screenshost every second, I think I would end up DoSing my server. It would be better if the extension could "feel" that latency and back off
  - I suppose a rudimentary way to do it would be to make all the request processing on the `/receive` endpoint synchronous, so it only sends a response after classification is finished and the switcher has switched (if needed). Of course, the extension's scheduled screenshot sending would need to be tweaked so it could tell if it had gotten a response from the server for its last request yet and skip sending a new screenshot if it hasn't.
- [ ] elegantly handle timeouts from both the llama.cpp server and the HDMI Matrix control server

- [/] review jpg support -- some places still assume png
- [ ] don't save compressed images into the same directory as their originals
- [ ] build an abstraction layer for accessing image files and associated data

- [ ] the receiver saves the received image as a file, but then `_classify_image` takes the file path and reads it as base64 -- maybe that can be cut out
- [ ] factor out an enum of classification labels to support more consistent typing
- [ ] keep track of last receive time -- if we haven't gotten a new screenshot in a while (depending on the receive frequency), update the state to reflect possible connection loss and show that in the UI as well

#### Review

- [ ] I keep wanting to add new label types or update existing ones -- e.g. now I want to just tag every image that's a Fox side-by-side ad break -- maybe support custom tags of some kind

#### Accuracy

- [ ] when saving images, associate them with the active prompt and classification setup (this seems hard)

- [ ] improve the data model of how inaccurate frames are saved. dumping them all in a folder with a .json file feels gross
- [ ] increase the number of recent images we retain when reporting an error

#### OpenCV

##### `logo_match.py`

- [ ] `LOGO_PATH` is currently hardcoded -- it should be configurable

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

- [ ] support switching between multiple prompt files

- [ ] for training purposes, record a level of "importance" associated with each manual classification -- how much do I care about this being classified correctly?
  - though this might also be better addressed by just improving my categories
  - the real issue is that I don't want to penalize the classifier too much for edge cases like transitions to and from commercial

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

#### Future ideas

- [ ] to improve the prompt further, grab closed captions/subtitles and include them with the screenshot when sending to the LLM
  - [ ] can we grab subtitles/captions from the `<video>` tag?
- [ ] somehow capture the broadcast audio and use whisper or something with speaker diarization to check if one of the Fox hosts is talking
  - or try using some kind of audio classification model
  - maybe volume and dynamic range would be a viable signal?

### Switching

- [ ] try to avoid sending multiple parallel/back-to-back requests to change inputs
- [ ] if I re-enable auto-switch, immediately switch to the right state for the current classification

### UI

- [ ] display the reason for the categorization on the UI
- [ ] maybe use a CSS framework
- [ ] mobile-friendly UI so I can use it on my phone (another argument for using a CSS framework -- something like Bootstrap would probably make this easier)
- [ ] seems like the `/is_ad` SSE channel gets disconnected if the server is down for too long (more than a few seconds) or if I SIGTERM it -- the page should detect if the connection is closed, show a "connection lost" message, and fall back on polling
- [ ] kinda wish I had hot reload on the frontend when I make UI changes
- [ ] right now, the client requests `/is_ad/last_frame?t=${Date.now}` every time it receives a message from the server, even if there's no new image. this should be updated to avoid a pointless fetch if the image hasn't changed
- [ ] move css out into a single style.css file and unify between the /review and /is_ad endpoints

- [ ] make the "Pending" state also change the background color (maybe to yellow, or a different shade of red/green depending on the classification)
- [ ] when in the "Pending" state, provide a way to preempt and tell it it's wrong
- [ ] when in the "Pending" state, provide a way to confirm it's right and switch right away

- [ ] when I click "Report", it should include a unique ID (timestamp?) of the image reported so there's no race condition from hitting it a split second too late -- currently, I think there's a race condition where I could hit "Report" just as it changes and it would associate that with the wrong image (though it retains multiple images, so maybe it's fine)
- [ ] ambitious: when I click "Report", temporarily update the classifier prompt to include the relevant screenshot as an example.
- [ ] when I tap "Report", show a popup with all the recently captured frames and their classifications. let me pick which ones specifically were classified wrongly and save the whole batch

- [ ] stretch: allow controlling YTTV (pause, rewind, etc.) from the web UI
- [ ] include `incorrect_frames` in the `/review` endpoint so I can classify them
- [ ] if the server stops responding when the UI polls for updates, show that the connection was lost
- [ ] show a counter of the number of seconds since the last image was received
- [ ] the "Report" button stays highlighted on my iPad after I've tapped it (I had to turn off the transition effect because it made the button flash every second)
  - I guess the button gets focused and then doesn't unfocus -- try unsetting the focus when I tap on the background and/or automatically after a delay
