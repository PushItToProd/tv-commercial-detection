browser.browserAction.onClicked.addListener(async (tab) => {
  try {
    // 1. Ask the content script for the video element's bounding rect
    const results = await browser.tabs.executeScript(tab.id, {
      file: "content_scripts/get_video_bounds.js"
    });
    const rect = results[0];

    // 2. Capture the full visible tab
    const dataUrl = await browser.tabs.captureVisibleTab(tab.windowId, {
      format: "png"
    });

    // 3. Crop to the video rect (or skip cropping if no video was found)
    const croppedUrl = rect
      ? await cropImage(dataUrl, rect)
      : dataUrl;

    if (!rect) {
      console.warn("No playing video found on this page — saving full screenshot.");
    }

    // 4. Convert to a blob URL (required by Firefox for downloads)
    const blob = dataUrlToBlob(croppedUrl);
    const blobUrl = URL.createObjectURL(blob);

    // 5. Download
    const title = (tab.title || "capture").replace(/[\\/:*?"<>|]/g, "_").trim();
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${title}_${timestamp}.png`;

    await browser.downloads.download({
      url: blobUrl,
      filename: filename,
      saveAs: false
    });

    console.log(`Frame saved as: ${filename}`, rect ?? "(full screenshot)");
  } catch (err) {
    console.error("Capture failed:", err);
  }
});

// ---------------------------------------------------------------------------

function cropImage(dataUrl, rect) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = rect.width;
      canvas.height = rect.height;
      canvas.getContext("2d").drawImage(
        img,
        rect.x, rect.y,         // source x, y
        rect.width, rect.height, // source w, h
        0, 0,                    // dest x, y
        rect.width, rect.height  // dest w, h
      );
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

function dataUrlToBlob(dataUrl) {
  const [header, data] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}