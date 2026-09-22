# Hyperframes Composition Brief: FinanceHQ

## Objective
Create a short launch-style brag video for FinanceHQ, a RAG system for loan-document Q&A.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080
- Duration: 20 seconds

## Source Material
- Project root: `/Users/Masters/Projects/FinanceHQ`
- Primary files read: `react_ui/src/index.css`, `react_ui/tailwind.config.js`, `react_ui/src/components/ui/animated-hero.tsx`, `react_ui/src/components/ui/Navbar.tsx`, `react_ui/src/components/overview/{StatsBar,ABResults,TerminalDemo,PipelineSteps}.tsx`, `react_ui/src/components/chat/{ChatPage,UploadDropzone,MessageBubble}.tsx`, `react_ui/src/components/ui/button.tsx`, `README.md`
- Product name: FinanceHQ
- Tagline / strongest claim: "Upload a loan PDF. Ask anything. Get document-grounded answers in real time with no hallucinations, no guessing."
- Key UI or visual moment to recreate: the TerminalDemo-style streaming window (macOS traffic-light dots, monospace SSE token stream, amber "streaming" pill, metadata tag chips) and the AB leaderboard row with its amber "Winner" badge
- Copy that must appear verbatim:
  - "Loan answers, grounded."
  - "What is the interest rate?"
  - "The interest rate on this loan is 8.65% per annum (floating), linked to the bank's MCLR..."
  - "3 LLMs. 1 winner."
  - "Document-grounded answers. Zero hallucinations."

## Creative Direction
- Tone preset: polished
- Creative direction: confident MLOps product demo; dark trading-terminal aesthetic; restrained motion, one real payoff beat
- Interpretation: fewer, longer-held scenes; no jokes or chaos; confidence through restraint, real UI fidelity, and one clean data-backed brag moment rather than exaggerated claims
- Angle: This is a hiring-manager-facing portfolio piece, not a joke product. Treat FinanceHQ like a trading terminal for loan documents — dark charcoal, amber accents, monospace streaming text — and let the real MLOps machinery (live SSE streaming, A/B testing, MLflow-tracked model routing) be the "wow" instead of invented hype.
- Hook: the site's own rotating hero word snaps from "instant" to "grounded" and holds — landing on "grounded" sets up the whole video's thesis before any UI appears.
- Outro / punchline: wordmark "Finance**HQ**" settles on black, under it: "Document-grounded answers. Zero hallucinations." — the thesis restated as fact.
- Avoid:
  - Generic SaaS language ("streamline your workflow," etc.)
  - Abstract filler visuals (no particle systems, waveform bars, stock motion graphics)
  - Any redesign of the product's real look — reuse its actual palette, type, and component shapes

## Visual Identity
- Background: `#0f0d0b` (near-black warm charcoal). The real site layers a subtle warm pixel grid (72px, `rgba(245,240,232,0.015)` lines) and a fractal-noise grain overlay (~3.2% opacity) on top — replicate this texture if feasible, it is core to the site's felt identity.
- Text: `#f5f0e8` (warm cream) at full opacity for primary text; `rgba(245,240,232,0.5–0.85)` for secondary/muted text
- Accent: `#f59e0b` (amber-500); accent glows/borders use `rgba(245,158,11,0.06–0.3)`
- Display font: Geist Variable (bold, tight tracking, large sizes for headlines) — bundled at `@fontsource-variable/geist`; fall back to a clean geometric sans if unavailable
- Body/mono font: Geist Mono for terminal/streaming text and tag chips — bundled at `@fontsource/geist-mono`; fall back to a monospace stack
- Visual references from the project:
  - Pill badge: rounded-full, amber border+bg-tint, small pulsing dot, uppercase tracked label — used for "RAG · Textract · FAISS · Groq SSE"
  - Terminal window: macOS traffic-light dots (`#ff5f57` `#febc2e` `#28c840`), thin bottom border, monospace label bar
  - Streaming answer text with a blinking amber caret cursor
  - Metadata tag chips: small rounded-md pills with a thin cream border at low opacity
  - AB leaderboard row: 4-column grid, amber-tinted background + amber "Winner" pill badge for the winning row
  - Button style: solid amber pill button with dark text for primary CTA feel (reference only, not required on screen)
  - Navbar wordmark treatment: "Finance" in cream + "HQ" in amber, bold tight tracking

## Storyboard
Use the full storyboard in `brag-output/brag-plan.md` as the creative contract. Scene summary:

1. **The claim** — 3s — Amber pill badge "RAG · Textract · FAISS · Groq SSE" appears, then giant headline "Loan answers, ___" shows "instant" briefly and snaps to "grounded", holding to the end of the scene. Must be readable: "grounded" needs a clear hold.
2. **The flow (centerpiece)** — 8s — Sub-beat A: a PDF ("home_loan.pdf") drops into the upload dropzone; a status row ticks PENDING → EXTRACTING → CHUNKING → READY. Sub-beat B: cut to a terminal-style chat window; question "What is the interest rate?" types out char-by-char; amber "streaming" pill appears; the real grounded answer streams token-by-token; metadata chips (`lookup_v2`, `1.24s`, `5 chunks`) pop in. This is the most important scene — it must show the actual product working, not a diagram of it.
3. **The flex** — 6s — Headline "3 LLMs. 1 winner." slams in. The AB leaderboard row for `Llama 3.3 70B` (latency `2.02s`, groundedness `87%`, traffic `100%`, amber "Winner" badge) slides in and locks. Two stat chips pop in after: `<1.8s avg latency`, `115+ tests passing`.
4. **Outro** — 3s — Cut to black/charcoal. Wordmark "Finance**HQ**" (cream + amber) settles center, then "Document-grounded answers. Zero hallucinations." fades up and holds.

## Audio
- Audio role: cinematic support — restrained, confident, never louder than the product
- Audio arc: steady clean bed under Scenes 1–2, a slight energy lift under Scene 3 (the flex/payoff beat), fade-out through Scene 4
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (109.96 BPM, "steady and clean")
- Music treatment: fade in over first ~0.4s, hold at 0.30–0.35 volume through Scenes 1–2, subtle lift in presence for Scene 3, fade to silence by the final frame of Scene 4
- Music cue guidance: bundled preset at `assets/music/cues/happy-beats-business-moves-vol-12-by-ende-dot-app.music-cues.json` (and matching `.md`). Suggested strong-cue locks (use ±0.15s tolerance, adjust if it hurts readability): **8.74s** — streaming pill / answer begins (Scene 2); **13.11s** — "Winner" badge locks in (Scene 3); **17.47s** — hard cut into outro (Scene 3→4). Full beat grid available in the preset for any smaller sequential snaps (e.g. metadata chip pops, stat chip pops).
- Audio-reactive treatment: subtle — the amber glow on the terminal window border and/or the hero headline's accent color may breathe slightly with music RMS. No waveform/equalizer visuals, no strobing.
- Audio-coupled moments:
  - Scene 1 — hero word swap ("instant" → "grounded") — soft switch/tick sound exactly on the swap
  - Scene 2 — file drop — soft drop sound on landing; typed question — randomized keyboard keypress ticks per character; streaming pill appears — align near 8.74s cue; metadata chips — soft pop on first and last chip only (not every chip)
  - Scene 3 — "Winner" badge lock — one restrained bell/impact hit, target ~13.11s cue; two stat chips — soft pop each
  - Scene 4 — wordmark settle — one soft low bell; music fade-out begins here
- SFX selection guidance: match the visible gesture — drop sound for the file landing, keypress sounds for typed text, chip/card-pop sounds for sequential UI elements, one bell/impact-style cue reserved for the single biggest payoff (the Winner badge). Keep density low — polished tone gets 3-4 SFX total, not one per element.
- SFX analysis guidance: consult the bundled `sfx-analysis.md`/`sfx-analysis.json` if present; prefer low/medium high-frequency-risk files for the repeated typing and chip-pop moments; the one bell cue can be a slightly more present, still-clean impact/bell family file.
- Exact SFX choice: Hyperframes should choose filenames, timestamps, density, and volume based on the implemented animation.
- Audio files: music copied to `brag-output/composition/assets/music/happy-beats-business-moves-vol-12-by-ende-dot-app.mp3`. Hyperframes should copy any chosen SFX into `brag-output/composition/assets/sfx/...` from the skill's SFX library.

## Hyperframes Instructions
Load the composition-building Hyperframes domain skills — `hyperframes-core` (composition contract + `data-*` timing), `hyperframes-animation` (motion), `hyperframes-creative` (design spec, beats, audio-reactive), `hyperframes-keyframes` (seek-safe keyframes), and `hyperframes-cli` (lint/check/render). `/brag` is its own workflow: do not enter the `hyperframes` entry-point intent interview and do not route into its generic promo / launch-video workflow. Prefer native Hyperframes conventions over anything in `/brag`.

Requirements:
- Show at least one real UI, copy, or visual element from the source project (the terminal streaming window and the AB leaderboard row both qualify — include both).
- Keep all text readable in the final render — respect the reading-time floors from `brag-plan.md` (short labels ~0.8s settled, full sentences ~0.3s/word).
- Keep the video within 15-25 seconds (target 20s).
- Include the planned music/SFX layer — audio was not disabled.
- Treat `/brag` audio notes as guidance, not a fixed cue sheet. Choose exact SFX files after the visual animation exists.
- Treat music cue metadata as optional timing hints. Ignore cues that hurt readability, scene pacing, or the product story.
- Major reveals may move toward nearby strong cues within about 0.15s. Smaller entrances may align to nearby beat points within about 0.10s. Use 1-3 strong cue locks total (the three listed above).
- Use SFX to support motion and interaction: drop/card sounds for the file/chip reveals, an announcement-style cue for the Winner payoff, key/click sounds for the typed question, restraint everywhere else.
- Honor the planned music treatment: fade-in, steady hold, lift for Scene 3, fade-out through Scene 4.
- Consider the Hyperframes audio-reactive workflow for a subtle glow/presence response on the terminal window border and/or hero accent color. Avoid waveform/equalizer visuals, musical-note graphics, generic particle systems, strobing, or heavy pulsing.
- Use local assets for audio and any required runtime/media dependencies.
- Run `hyperframes check` before render — it is brag's single gate.
