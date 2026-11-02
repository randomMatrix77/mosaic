# Mosaic v0.1 — Separate source graphics and icons

This pack contains **individually authored vector sources**, not screenshot crops. It accompanies the Mosaic v0.1 dark-purple multi-screen prototype.

## Contents

- `branding/svg/`: independent **Mosaic symbol**, wordmark, lockup, full app icon, and translucent welcome-wave background.
- `branding/png/`: PNG renderings of brand graphics at suitable export sizes.
- `icons/svg/`: independently usable **24×24, round-stroke icons** with white strokes; tint them to the UI token color.
- `icons/png_24`, `icons/png_48`, `icons/png_96`: ready-to-use icon PNGs at 1×/2×/4×.
- `android/vector_drawables/`: **Android VectorDrawable XML** versions of icons. Copy into `app/src/main/res/drawable/` and use `Icon(painterResource(R.drawable.mosaic_ic_send), contentDescription = ..., tint = ...)` in Compose.
- `illustrations/svg/` and `illustrations/png/`: reusable empty states for conversations, memories and indexed images.
- `design/tokens.json`: exact colors, gradients, radii, spacing and typography reference values.
- `design/MosaicColors.kt`: Compose color tokens.
- `design/asset-manifest.json`: mapping of icons to screens/components.
- `reference_photos/`: **illustrative imagery only** for prototype/mockups, not real indexed images. Do not prepopulate a user's gallery with them.

## Important implementation notes

1. **No microphone**. Voice/audio input is not supported in Mosaic v0.1. Do not add a microphone anywhere; the composer has image-attach, text-input, send/stop only.
2. Icons are not functionality. Hide actions that lack an actual handler. Do not add fake interactive skills, graph visualizations, image search, cloud modes or phone automation.
3. Reproduce the compositional feel (near-black surfaces, violet accents, rounded cards, subtle glow) without using screenshot images as a UI background.
4. For best rendering on Android, prefer the native VectorDrawable XML for interface icons; prefer high-resolution PNG for brand gradients, or import SVG gradient vector appropriately.
5. **App icon foreground** provided as illustration only. When publishing, create Android adaptive icon assets (foreground/background + monochrome) with the appropriate safe zone and launcher specifications.
6. `font-family` uses system Roboto; **no font files are bundled**. Use Android's installed Roboto fonts / Material Typography.
7. Brand mark and logos were redrawn as reusable scalable SVG assets to match the design *direction*; they are not guaranteed pixel-identical to earlier AI-generated images.

## Sample Compose use

```kotlin
Icon(
    painter = painterResource(id = R.drawable.mosaic_ic_send),
    contentDescription = "Send message",
    tint = MosaicColors.Primary,
    modifier = Modifier.size(24.dp)
)
```

Colors: background `#090B16`, surface `#141629`, elevated `#1B1B31`, primary `#AC7CFF`, body text `#F6F4FF`, secondary text `#ADB0CB`.
