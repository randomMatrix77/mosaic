#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
android_dir="$(cd "$script_dir/.." && pwd)"
adb_bin="${ADB:-/Users/kronos/Library/Android/sdk/platform-tools/adb}"

"$android_dir/gradlew" :app:assembleDebug
apk="$android_dir/app/build/outputs/apk/debug/app-debug.apk"
"$adb_bin" install -r "$apk"
"$adb_bin" logcat -d -s vibeOS-MNN:I AndroidRuntime:E '*:S'

