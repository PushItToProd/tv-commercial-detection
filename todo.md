# To Dos

Checkboxes key:

- [ ] to do
- [/] doing
- [-] canceled/abandoned
- [x] done

## Extension

- [ ] !! it has a terrible memory leak -- Firefox was using _gigs_ of RAM after I had it running for a few hours

- [ ] `track_interactions.js` registers event listeners on the video tag, but it doesn't unregister them if it later targets another video tag

- [ ] auto-save config when you click "start" (currently, it reports an error even though there's placeholder config pre-populated)
- [ ] notify the server when the sender extension is started or stopped (and show this on the UI)
- [ ] ensure we only accept images for classification from one tab/sender at a time -- don't want to accidentally DoS myself or confuse the classifier if I enable this on multiple tabs
- [ ] don't switch if I'm actively interacting with the video player tab (onmouseover?)
  - [ ] if I seek multiple times or catch up to live and end up on a commercial after having been on a break, wait to switch after a short delay in case I want to keep interacting
    - probably not that important of an enhancement compared to other options

- [ ] +if the classifier service returned its result to the extension, the extension could do things like mute and (if possible) skip ahead automatically until it's not on an ad break anymore
- [ ] package the extension so I can install it permanently in firefox

- [ ] consider doing some of the OpenCV processing in the extension itself? https://docs.opencv.org/4.x/d8/dd1/tutorial_js_template_matching.html
- [ ] could do some scene change detection or other CV stuff in the extension and use that to decide when to send images

- [ ] explore adopting a browser extension framework or scaffold of some kind - something to enable packaging, testing, etc.

## Web app

- [ ] update my config to move outputs (`frames`, outputs from `check_classification.py`, etc.) into a separate folder outside my local project directory

- [ ] persist state in a better way (SQLite? Redis? JSON file on disk?) than just keeping it in a dataclass in memory
- [ ] document expected values for `output_settings` in `AppConfig` (`config.py`)
- [ ] use `pydantic-settings` for settings -- https://docs.pydantic.dev/latest/concepts/pydantic_settings/

### Classification/Receiver

- [ ] parallelize processing -- send images to the LLM as fast as it can handle them but, while waiting for the response, keep processing incoming images with OpenCV in case they have a strong signal of an ad or not-ad
  - ~not fully sure this makes sense, but it's a thought -- I guess what I'm thinking is, suppose the broadcast is going to commercial and we receive a series of images like this:
    1. cars on track with scoreboard and network logo -- can be classified by OpenCV alone
    1. cars on track, now with no scoreboard or logo -- have to prompt the LLM to confirm it's racing. this can take 1-3 seconds
    1. some kind of interstitial bump to commercial (stage winner graphic or something)
    1. interstitial bump continues for several consecutive captures -- have to keep prompting the LLM and it flails indecisively between "ad" and "content"
    1. side-by-side transition graphic with the side-by-side boxes, Fox logo in the ad box, and no side-by-side scoreboard on screen yet (so we can't use logo matching) -- still prompting LLM
    1. finally, the side-by-side scoreboard appears with the logo
  - as soon as that side-by-side scoreboard appears, we should be able to instantly decide to make the switch without waiting any longer
- [ ] capture phashes of frames from the start of particularly annoying commercials and support changing as soon as they appear (likely using a pull-based model for this)

- [ ] I want a better way of tracking, end-to-end, how long it takes to switch after receiving an image. I also want a way to tell if things get backed up
- [ ] handle latency and backpressure -- right now, I have two external components that can have high-ish response times, but I have no way to handle that.
  - e.g. if I were to use a more intensive prompt for classification that takes >2s to run and then set the browser extension to send screenshost every second, I think I would end up DoSing my server. It would be better if the extension could "feel" that latency and back off
  - I suppose a rudimentary way to do it would be to make all the request processing on the `/receive` endpoint synchronous, so it only sends a response after classification is finished and the switcher has switched (if needed). Of course, the extension's scheduled screenshot sending would need to be tweaked so it could tell if it had gotten a response from the server for its last request yet and skip sending a new screenshot if it hasn't.
- [ ] elegantly handle timeouts from both the llama.cpp server and the HDMI Matrix control server

- [/] review jpg support one last time -- some places still assume png
- [ ] build an abstraction layer for accessing image files and associated data

- [ ] the receiver saves the received image as a file, but then `_classify_image` takes the file path and reads it as base64 -- maybe that can be cut out
- [ ] factor out an enum of classification labels to support more consistent typing
- [ ] keep track of last receive time -- if we haven't gotten a new screenshot in a while (depending on the receive frequency), update the state to reflect possible connection loss and show that in the UI as well

- [ ] allow importing classification profiles from a full Python path (so they could be added as plugins)
  - e.g. in `config.json` you could put `my_classifiers.some_sport` and it'd auto-import that package
- [ ] have an option to bypass the LLM-based classification in case I want to turn off the llama.cpp server temporarily

- [ ] can we take advantage of prompt caching to deduplicate prefill between the quick reject and full-match steps?
  - maybe just send the full-match prompt as a reply to the model's previous response? (except the model will have replied `yes` already, even if it's wrong, so that could influence the next reply).
  - alternatively, just structure the prompt to put the image and audio first maybe? (except then we lose the advantage of caching the long text prompt for the full match).
  - so maybe there's no good way to do this

#### Review

- [ ] I keep wanting to add new label types or update existing ones -- e.g. now I want to just tag every image that's a Fox side-by-side ad break -- maybe support custom tags of some kind
- [x] support better filtering of images based on classifications and things (like `view_classification_results.py`)
- [ ] I want like a timeline view that shows all the images captured in a given broadcast (or at least a given timeframe) in chronological order
- [ ] replace date range filters with custom ones that will use YYYY-MM-DD date formats

- [ ] for training/eval purposes, record a level of "importance" associated with each manual classification to use when evaluating results -- basically, how much do I care about this being classified correctly?
  - though this might also be better addressed by just improving my categories
  - the real issue is that I don't want to penalize the classifier too much for edge cases like transitions to and from commercial

#### Accuracy

- [ ] manually classify even more images to help my test cases
- [ ] when saving images, associate them with the active prompt and classification rules (this seems hard since it's hardcoded as a function right now)
- [ ] consider how to improve the data model of how inaccurate frames are saved. dumping them all in a folder with a .json file feels gross
- [ ] increase the number of recent images we retain when reporting an incorrect classification

#### OpenCV

- [ ] can we classify based on color grade to discriminate between race content and ads? the ads that confuse the model tend to have more "cinematic" color profiles
- [ ] the logo match gets confused when the upper right is mostly white

##### `logo_match.py`

- [ ] `LOGO_PATH` is currently hardcoded -- it should be configurable

#### Prompt

- [/] right now I have my prompt, image, etc. hardcoded for just Cup on Fox/FS1 -- I'll need to add separate prompts, logos, etc. for the other series and broadcasters too
  - [x] add a dropdown to the UI that lets you pick from multiple prompt presets
  - [ ] eventually: detect which series I'm watching using YTTV and/or live feed data (if a race is live)
  - series/networks to handle:
    - [/] O'Reilly on CW
      - they keep their "CW Sports" logo visible in the upper right during side-by-side, so `classify_image` has to be rewritten to handle that
      - the logo changes to yellow when it's a caution period
    - [x] Cup on Amazon Prime
    - [x] Cup on TNT
    - [x] Cup on NBC
    - [x] Trucks
  - [ ] update the prompt text

- [ ] support switching between multiple prompt files

- [ ] update the prompt to indicate it's likely an ad unless it has race cars?
  - I guess that's kinda what the first "quick reject" prompt is for
  - feels like this would probably exacerbate the current misclassifications

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
- [ ] capture an entire race broadcast (or multiple broadcasts) as frames+audio, then have Claude just iterate on ways to consistently and quickly detect ads -- let it churn overnight or w/e and see what it comes up with

#### Future ideas

- [ ] somehow capture the broadcast audio and use whisper or something with speaker diarization to check if one of the current network's hosts is talking
  - or try using some kind of audio classification model
  - maybe volume and dynamic range would be a viable signal?

### Switching

- [ ] try to avoid sending multiple parallel/back-to-back requests to change inputs
- [ ] if I re-enable auto-switch, immediately switch to the right state for the current classification

### UI

- [x] display the reason for the categorization on the UI
  - [ ] display the LLM's output
- [ ] seems like the `/is_ad` SSE channel gets disconnected if the server is down for too long (more than a few seconds) or if I SIGTERM it -- the page should detect if the connection is closed, show a "connection lost" message, and fall back on polling
- [ ] kinda wish I had hot reload on the frontend when I make UI changes
- [ ] right now, the client requests `/is_ad/last_frame?t=${Date.now}` every time it receives a message from the server, even if there's no new image. this should be updated to avoid a pointless fetch if the image hasn't changed
- [ ] move css out into a single style.css file and unify between the /review and /is_ad endpoints

- [ ] make the "Pending" state also change the background color (maybe to yellow, or a different shade of red/green depending on the classification)
- [ ] when in the "Pending" state, provide a way to preempt and tell it it's wrong
- [ ] when in the "Pending" state, provide a way to confirm it's right and switch right away

- [ ] when I click "Report", it should include a unique ID (timestamp?) of the image reported so there's no race condition from hitting it a split second too late -- currently, I think there's a race condition where I could hit "Report" just as it changes and it would associate that with the wrong image (though it retains multiple images, so maybe it's fine)
- [ ] ambitious: when I click "Report", temporarily update the classifier prompt to include the relevant screenshot as an example.

- [ ] stretch: allow controlling YTTV (pause, rewind, etc.) from the web UI
- [ ] include `incorrect_frames` in the `/review` endpoint so I can classify them
- [ ] if the server stops responding when the UI polls for updates, show that the connection was lost
- [ ] show a counter of the number of seconds since the last image was received
- [ ] the "Report" button stays highlighted on my iPad after I've tapped it (I had to turn off the transition effect because it made the button flash every second)
  - I guess the button gets focused and then doesn't unfocus -- try unsetting the focus when I tap on the background and/or automatically after a delay

## Additional tools

### `annotate_broadcasts.py`

- [ ] the server throws errors if a directory that existed when it was started gets deleted - have to restart to work around it
- [ ] the `({N}, {M} ruled)` value shown in the dropdown doesn't get updated until you restart the server
- [ ] enable searching by frame name
- [ ] allow querying on properties/facets/annotations
- [ ] the current facets/annotations setup is a little too hands-on and I'm pretty sure I'm not keeping things consistent. in particular, the `care_away` and `care_back` values are hard to keep consistent when evaluating lots of broadcasts over time. however, the fact that I'm thinking about consistency here means this should actually be able to be determined by policy -- a post-ad-break ad read should always have the same `care_away` and `care_back` values. -> I want to define some additional classification values beyond `ad` and `content`, initally for my use only, which would be used as sort of configurable presets. so I could pick from a list that includes "content: racing", "side-by-side", "full-screen ad", "ad read", "hype segment", etc., and picking "content: racing" would set default values of `video=live_race, audio=commentary, care_away=0, care_back=3, risk=safe, verdict=content`, while picking "side-by-side" would set `video=side_by_side, audio=spot_audio, care_away=3, care_back=0, risk=safe, verdict=ad` and "ad read" would set `video=live_race, audio=ad_read, care_away=2, care_back=1, risk=fraught, verdict=ad`. eventually this deeper taxonomy could also be used as its own set of classification labels but I'd have to think more about that
- [ ] data model - allow breaking up a broadcast into contiguous segments -- the regions I've currently marked by just adding notes to a bunch in bulk
- [ ] transcribe audio clips and show them in the UI

- [ ] add back the ability to view results of a classification run and compare with my classifications (like `review_ground_truth.py` did, but not in the messy way it did)
  - [ ] if I ctrl+shift+click on any image, select the entire contiguous range of images with the same classification as the selected image so I can quickly confirm the accuracy of a whole stretch. make sure this doesn't risk inadvertently overwriting any classifications i've already made -- if there's a stretch of say 50 images the model labeled as content but I marked the first 5 and the last 10 as ads, only select the 6th through 39th images of that stretch.

### `check_classification.py`

- [ ] write a reporting script that takes classification results and prints out min, P25, P50, mean, P75, P95, and max times taken grouped by model result
