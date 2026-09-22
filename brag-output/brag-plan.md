# Brag Plan: FinanceHQ

## What is this app?
FinanceHQ is a RAG system that reads a loan PDF (via AWS Textract), indexes it into FAISS, and answers natural-language questions about it in real time via Groq-streamed, document-grounded responses — with a live 3-model A/B test (tracked in MLflow) auto-routing traffic to the best-performing LLM.

## The angle
This is a hiring-manager-facing portfolio flex, not a joke product. The angle: treat it like a trading terminal for loan documents — dark charcoal, amber accents, monospace streaming text — and let the real MLOps machinery (A/B testing, MLflow, live SSE token streaming) be the "wow" instead of invented hype. The claim "no hallucinations, no guessing" pays off literally: we watch a grounded answer stream out of an actual document, then watch the system prove which LLM earned the traffic.

## Hook (first 2-3 seconds)
The site's own rotating hero word snaps from "instant" to "grounded" and holds. Landing on "grounded" (not "instant" or "streamed") sets up the entire video's thesis before a single UI screen appears.

## Key moments (the middle)
- The upload dropzone receiving a real PDF, ticking through pipeline status states fast, landing on READY.
- A real question typed into the chat, and the answer streaming token-by-token in the terminal-style UI, ending with its live metadata tags (variant, latency, chunks).
- The A/B leaderboard: three LLMs, one amber "Winner" badge, 100% live traffic — the concrete proof of the MLOps claim.

## Outro / punchline
Wordmark "Finance**HQ**" (cream + amber, matching the navbar treatment) settles on black, with the line "Document-grounded answers. Zero hallucinations." — the thesis restated as fact, not tagline.

## User flow worth showing
Entry → key action → result, pulled straight from the Chat tab:
1. Drop a PDF into the dropzone → session spins up, doc status ticks PENDING → EXTRACTING → CHUNKING → READY.
2. Type a real question ("What is the interest rate?") into the chat.
3. Watch the grounded answer stream in token-by-token (SSE), landing with metadata tags: intent, variant, latency, chunks used.

This is the centerpiece of the video (Scene 2) — the product actually answering a question beats any description of it.

## Tone
- Preset: polished
- Creative direction: confident MLOps product demo; dark trading-terminal aesthetic; restrained motion, one real payoff beat
- Interpretation: Fewer, longer-held scenes (4 total). No jokes, no chaos — confidence comes from restraint, real UI, and a clean data-backed brag rather than exaggeration. Motion is smooth and deliberate, not frantic.

## Format: landscape — 1920x1080
## Duration: 20s target

## Visual identity (from the project)
- Background: `#0f0d0b` (near-black charcoal, with a subtle warm grid + noise grain overlay)
- Accent: `#f59e0b` (amber-500), with `rgba(245,158,11,*)` used for borders/glows/badges throughout the real UI
- Text: `#f5f0e8` (warm cream), with `rgba(245,240,232,*)` at varying opacity for secondary text
- Display font: Geist Variable (bold, tight tracking on headlines)
- Body font: Geist Variable; monospace elements (terminal/streaming text, tags) use Geist Mono
- Strongest visual element: the TerminalDemo-style window (macOS traffic-light dots, monospace SSE token stream, amber "streaming" pill, metadata tag chips) — this is the product's own signature visual and should be recreated faithfully, not reinvented

## Share copy (draft)
Built a RAG system that reads loan PDFs and answers your questions in real time — then A/B tests 3 LLMs against each other and auto-routes traffic to whichever one actually earns it. 87% groundedness, <1.8s latency, zero hallucinations.

## Audio direction
- Role: cinematic support — restrained, confident, never louder than the product
- Music: `happy-beats-business-moves-vol-12-by-ende-dot-app.mp3` (109.96 BPM, "steady and clean" — matches polished/cinematic)
- Music treatment: fade in under Scene 1 (0–0.4s), steady bed through Scenes 1–2, a slight lift into Scene 3 (the flex beat), fade out through the last ~1s of Scene 4
- Music cue guidance: bundled preset read (`vol-12.music-cues.json`). Strong-cue locks: 8.74s (streaming pill / answer begins, in Scene 2), 13.11s (Winner badge locks in, Scene 3), 17.47s (hard cut into outro, Scene 3→4). Tolerances per policy (~±0.15s major reveals).
- Audio-reactive treatment: subtle — the amber glow on the terminal window border and the hero word may breathe slightly with music RMS. No waveform/equalizer visuals.
- SFX posture: minimal but present (polished tier) — 3-4 cues total, nothing aggressive
- Audio-coupled moments: soft tick on the hero word swap; keyboard keypress ticks on the typed question; a soft drop/pop on each metadata tag arriving; one restrained bell/impact hit when the "Winner" badge locks in
- Restraint rule: never stack more than one SFX at once; no glitch/chaotic families; music never exceeds 0.35 volume; SFX stay in the 0.55-0.65 range

## Storyboard

### Scene 1 — The claim — 3s
Recreate the site's hero: amber pill badge "RAG · Textract · FAISS · Groq SSE" fades/slides in at top. Below it, giant bold headline "Loan answers, ___" with the word slot showing "instant" briefly then snapping to "grounded" and holding. Background is the real `#0f0d0b` with the subtle warm grid + noise grain texture from the site.
Sequential/interaction: yes — pill badge appears first (0.0–0.3s), then headline appears with "instant" (0.3s), swaps to "grounded" at ~1.1s and holds to end of scene.
Audio intent: quiet, confident opener — music fades in under this scene.
Audio-coupled idea: one soft switch/tick sound exactly on the "instant" → "grounded" swap.
Music: steady bed fading in, low volume.
Transition mood: soft crossfade → Scene 2.

### Scene 2 — The flow (centerpiece) — 8s
Sub-beat A (0.0–3.0s in scene): Recreate the UploadDropzone. A file "home_loan.pdf" drops into the dashed amber-on-drag dropzone; a compact doc-status row appears and ticks quickly through PENDING → EXTRACTING → CHUNKING → READY (real status vocabulary from the app), landing on a green/amber READY state.
Sub-beat B (3.0–8.0s in scene): Cut/slide to the chat panel, styled like the real TerminalDemo window (traffic-light dots, "FinanceHQ — POST /query/stream" label). A question types out character-by-character: "What is the interest rate?" Then an amber "streaming" pill appears and the real grounded answer streams token-by-token: "The interest rate on this loan is 8.65% per annum (floating), linked to the bank's MCLR..." Hold long enough to read the full line. Finish with metadata tag chips popping in one by one: `lookup_v2` · `1.24s` · `5 chunks`.
Sequential/interaction: yes — file drop → status ticks one by one → typed question (char by char) → streamed answer tokens → metadata chips pop in sequence (first → last accented, not every token).
Audio intent: build the feeling of real, live software responding — mechanical, satisfying, culminating in quiet confirmation as the chips land.
Audio-coupled idea: soft drop sound on file landing; keyboard keypress ticks (randomized) on the typed question; the streaming-pill appearance aligns near the 8.74s music cue; soft pop on the first and last metadata chip only.
Music: steady bed, unchanged.
Transition mood: clean hard cut → Scene 3.

### Scene 3 — The flex — 6s
Headline slams in: "3 LLMs. 1 winner." Beneath it, recreate the AB leaderboard row for the winning model: `Llama 3.3 70B` · latency `2.02s` · groundedness `87%` · traffic `100%` with its amber "Winner" pill — sliding in and locking into place. Immediately after, two compact stat chips pop in beside it from the site's real stats: `<1.8s avg latency` and `115+ tests passing`.
Sequential/interaction: yes — headline first, then the winner row slides in and locks, then the two stat chips pop in one after the other.
Audio intent: this is the one moment allowed real punch — the payoff of the whole video.
Audio-coupled idea: one restrained bell/impact hit exactly as the "Winner" badge locks in (target ~13.11s music cue); soft chip-pop sounds for the two stat chips.
Music: slight lift in energy/presence under this scene only.
Transition mood: hard cut (aligned near the 17.47s music cue) → Scene 4.

### Scene 4 — Outro — 3s
Cut to clean black/charcoal. Wordmark "Finance**HQ**" centers and settles (Finance in cream `#f5f0e8`, HQ in amber `#f59e0b` — exactly matching the site's navbar treatment). Beneath it, the line "Document-grounded answers. Zero hallucinations." fades up and holds to the end.
Sequential/interaction: none — single clean settle, no further motion.
Audio intent: quiet, confident close.
Audio-coupled idea: one soft low bell on the wordmark's settle; music begins its fade-out at the start of this scene.
Music: fading out through the scene, silent by the final frame.
Transition mood: soft fade to end.

**Music mood for this video:** cinematic/polished — steady, clean, understated confidence throughout, with one lift for the flex beat.
**Audio summary:** A steady, restrained instrumental bed carries the whole video with a single energy lift at the A/B "Winner" reveal; SFX are sparse and functional (typing, drop, chip-pop, one bell) rather than decorative, reinforcing that this is real software working, not a highlight reel.
