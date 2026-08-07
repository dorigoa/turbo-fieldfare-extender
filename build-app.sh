#!/bin/bash
# Assemble TurboFieldfare.app from the SwiftPM release build.
# Place this script one directory below the repo root (e.g. scratch/ or Scripts/).
#
# Usage:
#   scratch/make-app.sh [output-directory]      # default: build/
#   scratch/make-app.sh --install               # also copy to /Applications
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
APP_NAME="TurboFieldfare"
BUNDLE_ID="com.turbofieldfare.mac"
VERSION="${TURBOFIELDFARE_VERSION:-0.1.0}"

INSTALL=0
OUT_DIR="$REPO_ROOT/build"
for arg in "$@"; do
    case "$arg" in
        --install) INSTALL=1 ;;
        *) OUT_DIR="$arg" ;;
    esac
done

APP="$OUT_DIR/$APP_NAME.app"
RELEASE="$REPO_ROOT/.build/release"

echo "==> Building release products"
cd "$REPO_ROOT"
swift build -c release --product TurboFieldfareMac
swift build -c release --product TurboFieldfareDecodeService

echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# The app resolves its decode service as a sibling of its own executable.
cp "$RELEASE/TurboFieldfareMac" "$APP/Contents/MacOS/"
cp "$RELEASE/TurboFieldfareDecodeService" "$APP/Contents/MacOS/"

# SwiftPM's generated Bundle.module accessor looks for resource bundles at
# Bundle.main.bundleURL, which for an .app is the bundle root itself -- not
# Contents/Resources. They have to live next to Contents/.
for bundle in "$RELEASE"/*.bundle; do
    [ -e "$bundle" ] || continue
    cp -R "$bundle" "$APP/"
done

echo "==> Generating icon"
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
SRC_ICON="$REPO_ROOT/Sources/TurboFieldfareApp/Mac/Resources/turbofieldfare-app-icon.png"
for size in 16 32 128 256 512; do
    sips -z $size $size "$SRC_ICON" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) "$SRC_ICON" \
        --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
rm -rf "$(dirname "$ICONSET")"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>              <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>       <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>        <string>$BUNDLE_ID</string>
    <key>CFBundleExecutable</key>        <string>TurboFieldfareMac</string>
    <key>CFBundleIconFile</key>          <string>AppIcon</string>
    <key>CFBundlePackageType</key>       <string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$VERSION</string>
    <key>CFBundleVersion</key>           <string>$VERSION</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key>    <string>26.0</string>
    <key>LSApplicationCategoryType</key> <string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key>   <true/>
    <key>NSSupportsAutomaticTermination</key><false/>
    <key>NSSupportsSuddenTermination</key><false/>
</dict>
</plist>
PLIST

# No codesign step: the Swift linker already ad-hoc signs both executables, and
# a bundle-level signature is impossible while the SwiftPM resource bundles sit
# in the bundle root -- codesign rejects that with "unsealed contents present in
# the bundle root". A locally built app carries no quarantine flag, so Gatekeeper
# never asks for one.
echo "==> Signature: $(codesign -dv "$APP/Contents/MacOS/TurboFieldfareDecodeService" 2>&1 | grep -m1 Signature)"

echo "==> Built $APP"

if [ "$INSTALL" -eq 1 ]; then
    echo "==> Installing to /Applications"
    rm -rf "/Applications/$APP_NAME.app"
    cp -R "$APP" "/Applications/"
    echo "==> Installed /Applications/$APP_NAME.app"
