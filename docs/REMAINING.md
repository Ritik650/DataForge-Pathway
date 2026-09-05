# What's left, and how to do it

The build is finished. Everything below is executed by **other people** — that
is the point, not a scheduling accident. Six review passes found five real
defects; the last two were in the verification machinery rather than the work.
When the checker starts producing the findings, author-adjacent review is
exhausted, and adding more built surface trades verified for unverified.

Three tasks. Then submit.

---

## 1. Cold-read — worth ~3 points

**Protocol.** Hand `docs/concept-summary.pdf` to someone who has not seen this
project. No preamble, no context, do not watch over their shoulder. One read.
Then ask for five things:

1. What is the claim?
2. What is the mechanism?
3. What does BDH do, and what does BDH-CQ do?
4. What is the evidence?
5. Name one limitation.

**Use two or three readers, not one.** One reader is an anecdote. Three is a
measurement, and this project measures things.

**Time them.** Seconds from finishing the read to stating the claim.

### Pre-authorised fixes

Act on these without re-deciding — they are already reasoned through.

| If they cannot state | Do this |
|---|---|
| BDH vs BDH-CQ as **separate systems** | Split that paragraph in two. Add explicitly: *BDH is published and we run it; BDH-CQ is proprietary and we do not.* |
| The **mechanism** | The Eq. (8) sentence is carrying too much load. Add one plain-language sentence before it. |
| The **evidence** | The control is not landing. Say *"we removed 2× more state from elsewhere and nothing broke"* in words, before the numbers. |
| A **limitation** | Move the periodic-bands finding earlier. It is the most concrete limitation in the document. |

After any edit: `python scripts/build_summary_pdf.py summary && python scripts/check_pdf.py`
— the word count is gated at 500–950 and will tell you if a fix pushed it over.

### If nobody fails

Put the result on the page, in the audience band:

> *Three readers, cold, no context: all stated the claim within N seconds.*

That converts an assertion about learning effectiveness into a measurement,
which is worth more than any fix would have been.

---

## 2. Phone — worth ~2 points

Open <https://ritik650.github.io/DataForge-Pathway/> on a real handset.

**The one judgement that matters:** the state heatmap keeps true 4×8px cells and
scrolls horizontally rather than shrinking. Does that feel usable, or broken?

- **Feels usable** → nothing to do. Record which device and viewport.
- **Feels broken** → add a labelled zoom inset, magnification factor stated,
  main view unchanged. **Do not rescale the heatmap.** A squashed heatmap
  misrepresents the sparsity, which is the entire point of showing it.

Also check: the verification badge renders and is green; the dose slider is
thumb-operable; nothing overflows horizontally except the heatmap, deliberately.

---

## 3. Hostile questioner — worth ~1 point

Twenty minutes with someone who has not seen it and is trying to break you.
The questions to invite:

- Find me a number on this page that isn't sourced.
- What happens if I set the dose to 128?
- How do you know this isn't just damage from removing state?
- Which parts are precomputed and which are live?
- Is the sparsity claim yours or the paper's?
- What did you get wrong?

You have answers to all of these. Being asked is different from having written
them down. Answers live in `docs/CLAIMS.md`, `docs/DESIGN_NOTES.md`, and the
page's own "what we did not get" section.

**If they find an unsourced number, that is a correctness finding.** It goes to
`scripts/verify_claims.py` as a new assertion, not to a manual edit — otherwise
the same gap reopens somewhere else.

---

## The point that cannot be bought

Roughly one point across criteria 1 and 3 is unreachable by any action.

Technical correctness caps below full because the central *mechanism* claim —
RoPE phase cancellation producing the periodic offset bands — is labelled a
hypothesis. It fits every observation and we did not prove it. Labelling it
honestly is what protects the other 24 points on that criterion.

Learning effectiveness is partly a judge's impression of the narrative.

Neither is a gap. Both are the cost of not overclaiming, which is what this
submission is built on.

---

## On the day

**Order matters.** Lead with the result, not the failures.

1. **The claim** — a fact just read lives in a small, locatable set of synapses.
2. **The live break** — click Targeted, watch p(answer) collapse.
3. **The control** — click Top-other: 2× more state removed, recall intact.
4. **The corrections log** — *when asked*, not before.

A submission that opens with its failures reads as apologising for itself. The
log is the best answer to "what did you get wrong" and most teams will not have
one — but it is an answer, not an opening.

---

## Before you submit

```
python verify.py          # 8 gates, ~10s, must be green
```

Then confirm: artifact URL opens without sign-in in a private window · repo is
public · both PDFs attached · README AI disclosure current · mentorship line
still accurate (currently: none).
