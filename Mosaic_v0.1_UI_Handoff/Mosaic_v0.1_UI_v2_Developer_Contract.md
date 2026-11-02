# Mosaic v0.1 — Developer-ready UI/UX Design Contract

**Design reference:** `screens/01_home.png` through `screens/10_fold.png` (separate PNGs).  
**Reusable source assets:** `Mosaic_v0.1_Assets/` (SVG, Android VectorDrawable XML, PNG, design tokens).  
**Interactive visual sample:** `prototype/index.html` (visual reference only; demo data and mocked clicks are not production logic).

## 0. Delivery rules and non-negotiable scope

Mosaic v0.1 is a **local-first Android AI app** running on a Samsung Galaxy Z Fold 6. Preserve existing Kotlin Compose → Chaquopy Python → JNI → llama.cpp architecture, Python SQLite chat and knowledge-graph state, EmbeddingGemma text/image embeddings, agent tools, skills and standalone Qwen vision. Avoid rebuilding working systems. The full-fledged UI implementation is pending, and Qwen vision within normal chat may require additional integration.

**Do not show a microphone or any audio actions.** No local audio capture, transcription or speech output exists. Hide unfinished PDF ingestion, generated apps, cloud APIs, phone automation and diffusion models. No fake tool toggles, fake data or invented “thinking” traces. Prototype data is illustration only.

The **screens are independent implementation references**; **do not** render the composite design-board picture as a screen. Use one shared component library and responsive layouts.

## 1. Exact design system

### 1.1 Color tokens (dark-first)

| Token | Hex | Usage |
|---|---|---|
| background | `#090B16` | Root window, gesture region |
| surface | `#141629` | Cards, drawer, source context |
| surfaceElevated | `#1B1B31` | Search fields, list rows, raised panels |
| surfaceActive | `#24203D` | Selected card, composer and interactive surfaces |
| border | `#302C48` | Dividers and card outlines |
| borderEmphasis | `#504268` | Focused controls and selected cards |
| primary | `#AC7CFF` | Icons, selected tabs, text accents |
| primaryDark | `#7546DC` | Dark end of CTA gradients |
| accentBlue | `#88A9FA` | Logo highlight, keyboard focus states |
| onBackground | `#F6F4FF` | Primary text |
| textSecondary | `#ADB0CB` | Explanations and metadata |
| textTertiary | `#777C9B` | Timestamps, placeholders |
| success | `#46D7A0` | Completed steps, installed/ready |
| warning | `#EAB974` | Limited and partial capabilities |
| error | `#FF809C` | Failed stages, irreversible actions |
| userBubble | `#503584` | User message fill |
| chatComposer | `#25233E` | Compose input surface |

Primary CTA gradient, **125°**: `#7440E3` → `#9B58EE` → `#C576FF`. Never apply the bright gradient to large background regions except the specific welcome illustration. Overall UI should remain mostly charcoal/near-black.

### 1.2 Typography

Use Android's installed **Roboto** (no custom font import). Display 28sp / bold 700–760, Section 21sp / 700, app bar 16sp / 700, standard body 14sp / 400–500, chat body 14sp with generous 1.45–1.6 line height, labels 12sp / semibold, support 11–12sp, micro metadata 10sp. Do not use tiny text for important actions. Honor font scaling and accessibility.

### 1.3 Layout, shape, elevation

8dp spacing grid, with 4dp adjustments. Horizontal cover-screen page margin **20dp**, chat thread 16–18dp. Cards radius **16dp**, large context cards **22dp**, icon buttons **14dp**, composer **26dp**, pills **20dp**, user bubble **18dp** with subtly tighter lower-right corner. Outlined card border **1dp**. Primary button **48dp** minimum height, composer **56dp** minimum height. All interactive targets **48×48dp** minimum. Small chip text never becomes a sub-48dp inaccessible target; pad hit areas separately. Use subtle Material3 elevation/shadows, never strong neon glow around every element.

### 1.4 Icon usage and assets

Use the **separate** icon assets under `Mosaic_v0.1_Assets/android/vector_drawables/mosaic_ic_*.xml`. Copy into Android `app/src/main/res/drawable` or translate to Compose ImageVectors. Display 24dp glyphs inside 44–48dp hit targets. Tint icons with primary/text colors at runtime. Do not use emojis as final icons. For the gradient brand use `branding/png/mosaic-symbol-256.png` and `mosaic-lockup-1024.png` as needed, or appropriately imported SVG. The translucent welcome graphic is `branding/svg/mosaic-welcome-wave.svg`. Empty illustrations are optional and only appear when actually empty.

**No microphone glyph, audio action or speech text anywhere.** In v0.1 composer, leading `+` opens image attachment action; trailing arrow **Send** changes to square **Stop** when generation begins.

### 1.5 Motion and responsiveness

Button press feedback: subtle ~120–180ms color/scale transition with reduced-motion accommodation. Expand agent activity smoothly, ~180–240ms; don't animate every stream token. Use a spinning progress indicator for active inference with **real elapsed time**, not fabricated progress %. Respect status/nav system bars and IME insets. Save scroll state and draft text when switching fold/cover display.

## 2. Information architecture

Primary areas: **Chat**, **Conversations**, **Memories**, **Images**, **Capabilities**, **Settings**. On compact cover screen, Chat is the default workspace; supporting areas open via the navigation drawer (or back navigation). On unfolded screen, use **left navigation rail/panel + central workspace + optional right context panel**. No separate top-level Knowledge Graph screen required; graph relationships are visible inside memory details.

**Navigation model:**
- `Home/New chat → Chat(chat_id)`;
- `Chat → Memories` through drawer; tapping a memory evidence source opens originating conversation *at the message anchor*, preserving the previous view on back;
- `Chat → Images`; selecting an existing image opens `Image detail`; 'Ask about photo' opens composer with chosen image attached;
- `Chat → Conversations`; a conversation tap selects its existing history;
- `Chat → Capabilities` displays actually available skills/tools as read-only capabilities;
- `Chat → Settings` shows actual backend and storage states.

## 3. Screen-by-screen specification

### Screen 01 — Welcome / empty Chat (`screens/01_home.png`)

**Purpose:** A welcoming, focused starting point before a chat is selected or when the conversation is empty.

**Visual:** Near-black background, brand mark/welcome wave in upper zone, one short headline, real local model status pill, and a large, rounded CTA. No oversized feature dashboard.

**Actions:** 'Start a conversation' creates a conversation using the existing Python persistence path, navigates to Chat, focuses message input on request. The `+` attachment affordance can open Android Photo Picker if user chooses. Optional chips can give grounded example prompts without invoking a model until tapped.

**States:** Model not installed → show clear import/load action; loading → show real loading state; ready → green on-device indicator; failed → retry and diagnostic link. Do not show “model ready” from hardcoded UI state.

### Screen 02 — Chat + streaming + agent activity (`screens/02_chat.png`)

**Purpose:** Primary daily experience. This screen must remain useful if Qwen takes 30–90 seconds.

**Layout:** 56dp top bar: menu (left), active conversation title (ellipsized), status pill or status shortcut, overflow action. Chat thread scrolls between top bar and **sticky bottom composer**. User messages align right in muted violet bubble; assistant response is left-aligned readable text with optional mildly elevated surface. Display inline citations as source chips below answer, not huge metadata blocks in body.

**Composer:** Leading `+` action (show available attachment types, presently **image only**); multiline text input grows up to ~5 lines; trailing **Send** enabled when valid text/image is selected, **Stop** while request running. Disable duplicate sends for same conversation unless concurrency is actually supported. Empty attachment preview removable with `x`. No microphone.

**Streaming:** Initialize assistant response placeholder only after request begins. Render **true** native token deltas; coalesce high-frequency updates to reduce recomposition. A partial `tool_call` or its JSON is *not* user-facing assistant text. Show final-answer stream only once agent has begun user-facing response. When done, finalize Markdown and source chips. Allow copy action for a completed message.

**Activity card:** Collapsed by default after a finished request; expanded during tool operations. Headline 'Working on your request', spinner and real elapsed time. Rows for model loading, prompt processing, `search_memories`, retrieved evidence, `search_images`, image inspection (when integrated), final generation. Completed steps green check; active violet ring; failure red warning. Never show hidden reasoning or fictional intermediate steps. If events lack details, show neutral 'Processing request locally'.

**Scroll policy:** Auto-follow only if viewport was near bottom when new delta arrived; freeze auto-scroll when user scrolls away; provide 'Jump to latest' pill. Restore the prior scroll position when navigating away/back. Keyboard must not cover composer.

**Errors and cancellation:** Stop must invoke actual cancellation, not just hide spinner. If partial response exists, label 'Stopped' or 'Interrupted' and do not mark it as successfully completed in SQLite. A tool failure should be shown with retry where safe, and may be followed by an honest partial answer. If app resumes after interruption, show explicit interrupted state. No silent continuation.

### Screen 03 — Conversations (`screens/03_conversations.png`)

**Purpose:** Find and manage persisted chat threads.

**Visual:** App bar + search field, chronological list rows with icon, title, last-message preview if available, relative timestamp. New-chat action with violet plus.

**Interactions:** Row opens corresponding chat; long-press or overflow opens Rename, Delete. Rename uses prefilled text and validates nonblank title. Delete requires confirmation, performs the actual SQLite cascade, offers undo only if supported by storage. Search filters locally; empty results show `empty-conversations` illustration and a simple CTA. If a background generation belongs to another conversation, its progress remains tied to the originating request/chat ID; never leak deltas across threads.

### Screen 04 — Memories (`screens/04_memories.png`)

**Purpose:** Browse evidence-backed knowledge graph claims, not generic chat summaries.

**Visual:** Search field, optional chips `All`, `Current`, `Tentative`, `Historical`, `Projects`, `Preferences` only when corresponding filtering exists. Memory rows display claim summary, entity/type, status chip, and source-count if available. Do not invent confidence percentages.

**Interactions:** Search invokes actual memory query, local SQL or semantic retrieval; show retrieval loading and empty results. Tap row opens Screen 05. Edits are never performed by changing a visible label alone. Empty state explicitly says no extracted memory available, with a bounded 'Index conversation' action **only if implemented**; otherwise omit.

### Screen 05 — Memory detail + evidence (`screens/05_memory_detail.png`)

**Purpose:** Explain *why* Mosaic knows a fact.

**Visual:** Claim title at top; small status/type chips; segmented subview `Details / Evidence` (not entirely different screens). Detail lists subject/predicate/object, related entities, whether current vs superseded, optionally revision history. Evidence lists original conversation title, message IDs, timestamps and exact source quote (resolved from SQLite). A source card has 'View original conversation'.

**Interactions:** Tap evidence → navigate directly to original conversation at anchored message and highlight it temporarily. Back returns to same memory and tab. Multiple evidence spans show as separate rows grouped by conversation. Use original message text, not synthesized quotation. If source was deleted or spans invalid, show 'Source no longer available'. Correct/Delete actions exist only when corresponding Python APIs are available; destructive action requires confirmation.

### Screen 06 — Indexed images / ask about a photo (`screens/06_images.png`)

**Purpose:** Discover images the user explicitly indexed and launch vision on a selected photo.

**Layout:** Two tabs: `Indexed` and `Ask about a photo`. Indexed shows a search bar and responsive thumbnail grid (2 columns compact; 3–4 wide). Separate cards for relevant source metadata/results. Ask tab has large Photo Picker CTA and prompt field; optionally route to existing Chat composer to avoid duplicated generation state.

**Interactions:** Search invokes genuine `images.search` and displays ranked real indexed results; don't prepopulate sample cat/coffee photos except in mockups. Tap thumbnail → Screen 07. Photo Picker uses Android system picker, obtaining permitted URI and copying bytes to app-private cache if needed. Attachments are scoped to user-approved selection and should be removable. Permission denial → clear recovery message. Search text and images have distinct loading states.

### Screen 07 — Image detail / vision (`screens/07_image_detail.png`)

**Purpose:** Preview a source image and ask about its actual pixels.

**Visual:** Large aspect-ratio-preserving image, filename/source, indexing state, `Ask about this image` CTA; optional full-screen pan/zoom if implemented.

**Interactions:** CTA attaches stable approved source ID to a chat or opens a vision prompt, invokes the verified JNI vision inference path once integrated, and displays response in Chat with real streaming/status. Do not claim vision succeeded when only an image-vector similarity score exists. Never pass arbitrary file paths from Qwen tool output to vision; resolve approved source IDs in Python. If normal-chat vision is unfinished, hide CTA until implemented or mark as limited diagnostics outside release UI.

### Screen 08 — Capabilities (`screens/08_capabilities.png`)

**Purpose:** Tell users what Mosaic can *actually* do; prevent overpromising and make tools discoverable.

**Visual:** List of capability cards: memory retrieval, image retrieval, image understanding, memory organization. Each card includes a simple outline icon, short explanation and honest status `Available` or `Limited`.

**Interactions:** Read-only information; optional tap can open relevant task if implemented. **Do not show checked skill toggles** unless toggling genuinely changes allowed tools/skills and persists. No fake 'Coder', 'Voice', 'Generated apps' or phone automation controls for v0.1.

### Screen 09 — Settings (`screens/09_settings.png`)

**Purpose:** Provide operational transparency without exposing fragile implementation controls.

**Visual:** Grouped cards for Qwen model status, EmbeddingGemma status, inference backend, theme, privacy/storage; version. Icon buttons secondary, local status green.

**Interactions:** Model load/import only when working; display real model ready/unavailable errors; theme dark/light if implemented; clear data strictly behind destructive confirmation and only if API supports it. If a control isn't wired to Android/Python state, render informational text, not an active switch. No cloud switch (not implemented).

### Screen 10 — Unfolded Fold 6 (`screens/10_fold.png`)

**Purpose:** Reuse screens in an adaptive wider composition, not a second app.

**Adaptive rule:** Use window size classes / actual available dp width rather than hardcoding device identity. Compact = single Chat pane with drawer; medium = Chat + optional collapsible navigation; expanded = persistent left navigation (~220–248dp), flexible center Chat (min 360dp), right contextual rail (~260–320dp) when there is enough space. Right context rail switches between Activity and Sources; only show source list if real references are available.

**Interactions:** Selecting chats changes center thread; selecting an evidence chip fills right pane (or navigates to memory detail on compact). Swapping activity/sources never interrupts inference. On folding/unfolding preserve active conversation ID, partial response, drafts, scroll anchor, expanded activity state, and selected source. Handle fold hinge/occlusion through WindowManager where appropriate.

## 4. Reusable Compose component specifications

| Component | Shape / visual style | Interaction contract |
|---|---|---|
| `MosaicTopBar` | 56dp height, transparent background, icon targets 48dp | Menu/back, truncated title, status, overflow |
| `MosaicPrimaryButton` | 48dp high, 16dp radius, violet gradient; white label | Disabled/pressed/loading clearly differentiated |
| `MosaicIconButton` | 48dp target, 24dp icon, 14dp hover/pressed fill | contentDescription always set |
| `MosaicCard` | `surface` / `surfaceElevated`, 1dp `border`, 16dp radius | Optionally clickable with border-emphasis state |
| `MosaicStatusPill` | 20dp radius, green dot for ready, no gradients | Value from actual model lifecycle events |
| `MosaicChip` | 20dp radius; selected = darker violet fill | Selected state and text contrast, 48dp hit area |
| `MosaicComposer` | 56dp min, 26dp radius, `chatComposer` fill | Text, image attachment, send or stop, IME-safe |
| `MosaicMessageBubble` | Right user violet, left assistant transparent/card | Copy/selection; Markdown only after safe finalization |
| `MosaicAgentTimeline` | Elevated low-glow panel, active violet/completed green | Expand/collapse; real stage changes only |
| `MosaicMemoryRow` | 16dp rounded raised row, entity icon and status | Opens correct claim detail |
| `MosaicEvidenceCard` | Slightly higher contrast, original quote inset | Opens source message anchored in chat |
| `MosaicImageTile` | Rounded-clip image with optional label overlay | Opens real image detail with stable source ID |
| `MosaicErrorInline` | Compact red-accent surface | Shows actual error + retry when safe |

## 5. Contracts from Python to Compose

Do not hardcode string-based guessing in UI based on model text. Use structured, versionable request events with a stable `requestId`, `conversationId` and monotonic sequence.

```json
{
  "schemaVersion": 1,
  "requestId": "request-123",
  "conversationId": "conversation-xyz",
  "sequence": 8,
  "type": "tool_completed",
  "timestampMs": 1772345200000,
  "payload": { "tool": "search_memories", "displayName": "Search personal memories", "resultCount": 4 }
}
```

Expected event names: `request_started`, `model_loading`, `model_ready`, `prompt_processing`, `tool_started`, `tool_completed`, `tool_failed`, `answer_started`, `answer_delta`, `answer_completed`, `request_cancelled`, `request_failed`. When tool calls require model generation first, avoid claiming the model is idle while it generates a tool command; show neutral `Processing request locally` until the action starts.

**State rules:**
- Per-request state = `idle | preparing | tools | streaming | complete | cancelled | failed`.
- `answer_delta` only appends to matching request's in-memory partial assistant text.
- `answer_completed` triggers final persistence and final Markdown layout.
- A `tool_completed` never means the whole request finished.
- Model switching visible as actual `model_loading`/`model_ready` stages.
- Preserve all existing message persistence guarantees; interrupted partial messages cannot masquerade as successful assistant turns.
- Native callback → Python and Kotlin ViewModel must be nonblocking and main-thread safe; coalesce high-frequency deltas and avoid full-thread recomposition on every token.

## 6. Error, privacy and accessibility checklist

- No microphone, audio recording, audio mode or unsupported input affordance.
- AI 'thoughts' = **observable agent operations only**, not private reasoning text or a fake stream.
- A 90s request can show elapsed time and current real stage without fabricating percentages.
- Back navigation must not silently lose drafts; stop/cancel must actually cancel native work.
- Keyboard must never cover composer; unfolded screen must not obscure controls at hinge.
- Photo Picker only; never scan gallery automatically.
- Memory source evidence must be sourced from original stored messages, not generated fake quotes.
- Deleting a memory must respect no-recreation safeguards; deleting a conversation must handle linked evidence according to persistence policy.
- At minimum 48dp tap targets, readable high-contrast labels, TalkBack descriptions and dynamic text scaling.
- Show empty, loading, error, unavailable and success states for every data-driven screen.

## 7. Suggested implementation order for Codex

1. Import palette tokens, icon assets and reusable Compose controls; match `screens/02_chat.png` frame.
2. Implement streaming/event state in Python ↔ Kotlin ViewModel, Send/Stop, lifecycle and persistence.
3. Chat + Conversations with navigation and actual SQLite-backed data.
4. Memories list + evidence detail + anchored source-message navigation.
5. Images list + Photo Picker + image detail and real Qwen vision in Chat.
6. Capabilities/settings honesty and Fold 6 adaptive composition.
7. Polish states, accessibility, cancellation, theme and visual fidelity.

**Device availability:** Compile and test on Mac first. Do not request ADB or block waiting for Fold 6 during implementation. After tests and APK compile pass, report exactly what remains to verify, ask for physical connection, then run real device tests. Do not clear app data.

## 8. Visual implementation acceptance checks

- Screenshot each screen on compact cover and unfolded configurations and compare against corresponding PNG at the same scale. Check color, spacing, corner treatment and text density—not arbitrary pixel-perfect equality.
- Confirm at least `background #090B16`, `primary #AC7CFF`, `surface #141629` and primary button gradient actually applied.
- Icons must use individually supplied VectorDrawables or theme-matching ImageVectors, not emoji or screenshots.
- No microphone button, no fake capabilities, no placeholder user photo grid in production.
- Chat displays a *real* agent timeline and *real* streamed answer; Stop can cancel work safely.
- Source badges navigate to precise conversation evidence where supported.
- Existing inference, embeddings, memory, tools and ADB debugging must remain operational.

This is an implementation contract for the real v0.1 product, while screen PNGs communicate desired look and component hierarchy. Do not use the HTML prototype's hardcoded sample events or sample memory/image data in the Android app.
