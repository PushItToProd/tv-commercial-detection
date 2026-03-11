browser.browserAction.onClicked.addListener(async (tab) => {
  console.log("Browser action clicked, capturing tab...");
  try {
    // Capture the visible area of the current tab as a base64 PNG
    const dataUrl = await browser.tabs.captureVisibleTab(tab.windowId, {
      format: "png"
    });

    // Build a filename from the page title + timestamp
    const title = (tab.title || "capture").replace(/[\\/:*?"<>|]/g, "_").trim();
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `${title}_${timestamp}.png`;

    // Convert data URL to blob URL (downloads.download() doesn't accept data URLs)
    const response = await fetch(dataUrl);
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);

    // Trigger a download
    await browser.downloads.download({
      url: blobUrl,
      filename: filename,
      saveAs: true   // set to true if you want a Save dialog every time
    });

    console.log(`Frame saved as: ${filename}`);
  } catch (err) {
    console.error("Capture failed:", err);
  }
});
