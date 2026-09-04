# UI spec — paste into Stitch

Copy any block below. Section 1 alone is enough for a first screen; sections
2–9 are per-component if you want to generate them separately.

Reference design canvas: two artboards, full page + component sheet.

---

## 0. One-paragraph version (shortest useful prompt)

> A single-page interactive technical explainer titled "Where a fact lives",
> subtitle "Reading and breaking the synaptic state of Dragon Hatchling".
> Distill.pub style: dense, scientific, light warm background #fbfaf7, ink
> #1a1a18, one rust accent #b3401b, one teal #2f6f5e for passed controls.
> Source Serif 4 for prose, IBM Plex Mono for every number and label.
> Square corners everywhere — no rounded pills, no drop shadows, no gradients.
> Sections: header with a green verification badge; a boxed claim sentence; a
> row of token chips; a wide dense heatmap; a controls row with a segmented
> button group and a discrete slider; prediction bars beside a ground-truth
> box; two small charts; and a muted panel of withdrawn results.

---

## 1. Design system

| Token | Value | Use |
|---|---|---|
| ground | `#fbfaf7` | page background, warm — never pure white |
| surface | `#ffffff` | panel interiors |
| ink | `#1a1a18` | headings, primary text |
| prose | `#55544d` | body copy |
| muted | `#6b6a63` | captions |
| faint | `#85847c` / `#a8a79f` | axis labels, units |
| rule | `#e3e0d8` | dividers |
| border | `#d5d2c8` | panel borders |
| **accent** | `#b3401b` | the thing under test / targeted / danger |
| **pass** | `#2f6f5e` | control survived / verified |
| accent wash | `#f6e6de` | highlighted binding background |
| pass wash | `#f2f8f4` | verification badge background |

**Type.** Source Serif 4 — 52px/700 title, 22px/600 section heads, 27px claim,
16px/1.55 body. IBM Plex Mono with tabular figures — every numeral, token,
axis label, and uppercase 11px tracking-0.11em section label.

**Rules.** Square corners (border-radius: 0) everywhere. 1px borders, no
shadows, no gradients except the one heatmap legend ramp. Flex/grid with `gap`,
never margin-spaced siblings. Page padding 96px left/right.

---

## 2. Header

Left: an uppercase mono eyebrow "SYNAPTIC PLASTICITY AS SHORT-TERM MEMORY",
then the 52px title, then a 20px subtitle.

Right, top-aligned: a **verification badge** — 1px border `#cfe0d6`, background
`#f2f8f4`, a small checkmark SVG stroked `#2f6f5e`, and mono 12px text
"forward pass verified · max|diff| 2.6e-6". Below it, mono 11.5px faint:
"27,776 params · 8,192 state entries · 1.9 ms/forward".

A failure variant of the badge also exists: border `#e3c9c0`, background
`#fdf2ee`, an ✕ SVG, text "forward pass MISMATCH — results not trustworthy".

---

## 3. Claim block

Full-width, left border 3px `#b3401b`, 26px left padding, no background.
27px text, max-width 1000px. One phrase inside gets a `#f6e6de` highlight.

---

## 4. Token chips (the sequence row)

A wrapping flex row, 7px gap. Each chip is mono 13.5px, 7px/11px padding,
1px border, with a tiny faint position number underneath.

Four states:

| State | Background | Border | Text |
|---|---|---|---|
| binding under test | `#f6e6de` | `#b3401b` | ink, 600 |
| competing binding | `#eeece5` | `#d5d2c8` | `#3d3b34`, 400 |
| filler (binds nothing) | `#f6f5f1` | `#cbc9c0` **dashed** | `#a8a79f`, 400 |
| the query | `#1a1a18` | `#1a1a18` | `#fbfaf7`, 600 |

Below the row, a legend of swatch + label pairs, and a right-aligned mono
caption: "8 bindings · 2 filler · querying the oldest · offset 19".

---

## 5. State heatmap (the centrepiece)

A white panel, 1px `#d5d2c8` border, 16px padding, containing a **1024 × 256px
canvas** — 256 columns × 32 rows of 4×8px cells, `image-rendering: pixelated`.

Cell colour ramps warm-white → near-black by magnitude, mostly pale (the state
is sparse). Eight scattered cells are knocked out to white and ringed in the
accent — these are "the entries this binding wrote".

Below: mono caption on the left, and on the right a magnitude legend — a 96×9px
gradient strip `#faf9f6 → #d8d3c4 → #8f8a76 → #2c2a22` between "low" and "high".

To its right, a 250px column: an accent-bordered callout on `#fdf6f2`, and a
muted note.

---

## 6. Controls (two only)

**Segmented control.** Five square buttons — None / Targeted / Matched /
Top-other / Random — 1px `#d5d2c8` borders with `margin-left: -1px` so borders
merge. Selected inverts to `#1a1a18` background, `#fbfaf7` text, 600 weight.

**Discrete slider.** A 2px `#e3e0d8` track with the portion up to the active
stop filled `#b3401b`. Seven stops labelled 2 4 8 16 32 64 128; inactive stops
are 9px hollow circles with `#d5d2c8` borders, the active stop is a 15px filled
accent circle. Above it, a label row: uppercase "DOSE" left, mono
"8 entries · 0.098% of state" right.

Under a divider, two stat blocks side by side: "STATE MASS REMOVED 1.00×
targeted" in accent, "CONTROL REMOVES 2.06× top-other, and survives" in teal.

---

## 7. Prediction vs ground truth

One white panel split by a 1px left rule.

Left: five horizontal bars. Each row is `[mono token, right-aligned, 62px] ·
[20px track on #f4f2ed with a filled portion] · [mono value, 48px]`. The top
bar is accent when ablated, teal when intact; the rest `#cfccc2`. Below the
bars, a 34px mono readout "0.614" with "p(Oslo) ↓ from 0.960" beside it.

Right, 176px: uppercase "GROUND TRUTH", then a **2px solid `#1a1a18` box**
containing one mono 25px word, then a muted explanation. Truth sits *beside*
the estimate, never below it.

Under the panel, a row of mono stats with confidence intervals.

---

## 8. Two charts, side by side

**Dose–response.** SVG line chart, 580×230. Three polylines: targeted (accent,
2.5px, falls steeply), top-other control (teal, 2.5px, stays high), random
(`#c9c6bb`, 2px, flat). Dots at each measured point. Horizontal gridlines
`#f0eee8` at 0/50/100%, mono axis labels, x-axis "entries removed".

**Periodic bands.** SVG bar chart, same size. Eleven bars at token offsets
1,3,5,7,9,11,13,15,17,19,21. Bars below 90% are accent-stroked on `#f6e6de`;
the rest are `#cfccc2` on `#eeece5`. Undersampled bars (offsets 19, 21) use a
`3 2` **dashed** stroke. Caption: "dashed = 3 cells or fewer · shape beyond 17
unresolved".

Both charts: 1px panel border, 17px/600 title, 14px muted one-line subtitle.

---

## 9. "What we did not get"

Three equal cards in a grid, background `#f6f5f1`, border `#ddd9ce`, 17px
padding. Each: a 16px/600 title with a small uppercase mono status tag pushed
right (1px `#c4bfb2` border, `#77756c` text), then 14.5px body.

Tag vocabulary: `measured` (teal), `precomputed` (grey), `hypothesis` (accent),
`withdrawn` (grey), `unresolved` (grey).

---

## 10. Things to tell Stitch *not* to do

- No rounded corners, pill buttons, drop shadows, or gradient backgrounds.
- No emoji, no icon fonts — inline stroke SVG only, 24px grid.
- No hero image, no marketing copy, no call-to-action.
- Don't centre the body text; this is a left-aligned document.
- Don't use Inter or Roboto.
- Numbers never in the serif face.
