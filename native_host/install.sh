#!/usr/bin/env bash
# Install the TV Commercial Detector audio capture native messaging host.
#
# Run once after cloning the repo:
#   cd native_host && ./install.sh
#
# Re-run after moving the repo to update the absolute path in the manifest.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_SRC="$SCRIPT_DIR/com.tvdetector.audio_capture.json"
HOST_SCRIPT="$SCRIPT_DIR/audio_capture.py"
DEST_DIR="$HOME/.mozilla/native-messaging-hosts"
DEST_MANIFEST="$DEST_DIR/com.tvdetector.audio_capture.json"

# Ensure the host script is executable
chmod +x "$HOST_SCRIPT"

# Ensure Python and sounddevice are available
if ! command -v python3 &>/dev/null; then
  echo "Error: python3 not found. Please install Python 3." >&2
  exit 1
fi
if ! python3 -c "import sounddevice" 2>/dev/null; then
  echo "sounddevice not found — installing via pip..."
  # FIXME: use a dedicated virtual environment instead of --user, to avoid
  # conflicts with other Python packages
  python3 -m pip install --user sounddevice numpy
fi

# Create the native-messaging-hosts directory if needed
mkdir -p "$DEST_DIR"

# Write the manifest with the actual absolute path substituted in
sed "s|__INSTALL_DIR__|$SCRIPT_DIR|g" "$MANIFEST_SRC" > "$DEST_MANIFEST"

echo "Installed: $DEST_MANIFEST"
echo "Host script: $HOST_SCRIPT"
echo ""
echo "Reload the extension in Firefox (about:debugging) for the change to take effect."
