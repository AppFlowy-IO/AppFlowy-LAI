#!/bin/bash
set -e

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

echo "🔍 Checking for required tools..."
for cmd in curl jq unzip; do
  if ! command -v $cmd &> /dev/null; then
    echo "❌ $cmd is required but not installed. Please install it first."
    case $cmd in
      jq)
        echo "   You can install it with: sudo apt install jq"
        ;;
      unzip)
        echo "   You can install it with: sudo apt install unzip"
        ;;
    esac
    exit 1
  fi
done

# Create temporary directory
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "📥 Fetching latest release information..."
RELEASE_INFO=$(curl -s "https://api.github.com/repos/AppFlowy-IO/AppFlowy-LAI/releases/latest")

# Get tag name for version info
TAG_NAME=$(echo "$RELEASE_INFO" | jq -r '.tag_name')
echo "📦 Latest version: $TAG_NAME"

# Find the Linux asset
LINUX_ASSET_URL=$(echo "$RELEASE_INFO" | jq -r '.assets[] | select(.name | contains("Linux")) | .browser_download_url')

if [ -z "$LINUX_ASSET_URL" ]; then
  echo "❌ Could not find Linux release asset"
  exit 1
fi

echo "⬇️ Downloading AppFlowy LAI for Linux..."
curl -L "$LINUX_ASSET_URL" -o appflowy_lai.zip

echo "📂 Extracting files..."
unzip -q appflowy_lai.zip

# Find the binary in the extracted files
BINARY_PATH=$(find . -name "af_ollama_plugin" -type f | head -n 1)

if [ -z "$BINARY_PATH" ]; then
  echo "❌ Could not find the af_ollama_plugin binary in the extracted files"
  # List contents to debug
  echo "Contents of the extracted archive:"
  find . -type f | sort
  exit 1
fi

echo "🔧 Installing af_ollama_plugin to /usr/local/bin..."
mkdir -p /usr/local/bin
cp "$BINARY_PATH" /usr/local/bin/af_ollama_plugin
chmod +x /usr/local/bin/af_ollama_plugin

# Clean up
cd - > /dev/null
rm -rf "$TMP_DIR"

echo "✅ AppFlowy LAI plugin v$TAG_NAME successfully installed!"
echo "You can now use it by running: af_ollama_plugin"