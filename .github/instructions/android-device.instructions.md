---
description: Physical-device testing guidance for Mosaic Android sources.
applyTo: 'app/**'
---

# Android device testing

The Galaxy Z Fold can expose more than one display to ADB. Before automating Compose UI interactions, inspect `dumpsys display`; use the display ID reported by the UI hierarchy and pass it explicitly to both `uiautomator dump --display` and `input -d`. Do not assume that the default motion-event display is the visible Fold panel.

For persistence verification, install with `adb install -r` only: never clear app data or uninstall. A SQLite database in WAL mode may require copying the `-wal` and `-shm` sidecars alongside the main file before inspecting it on the development machine.
