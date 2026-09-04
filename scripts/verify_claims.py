"""Assert that every externally-sourced figure on the page is backed by CLAIMS.md.

The measured figures on the page bind from claims.js at runtime, so they cannot
drift from the JSON that produced them. The externally-cited ones -- BDH-CQ's
ARC results, the composition counts, the sparsity figure -- are HTML literals.
They are correct today because they were checked by hand against the papers.
Nothing stopped them drifting tomorrow. This closes that.

Two directions, because either failure is silent:

  LITERALS -> CLAIMS.md   every number written into the page must appear in the
                          claims ledger, so a typo or a half-remembered edit is
                          caught rather than shipped

  CLAIMS.md -> LITERALS   figures the ledger marks as load-bearing for the page
                          must actually appear on it, so a claim cannot quietly
                          vanish from the artifact while staying in the docs

Run: python scripts/verify_claims.py
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "artifact" / "index.html"
CLAIMS = ROOT / "docs" / "CLAIMS.md"
SUMMARY = ROOT / "docs" / "CONCEPT_SUMMARY.md"
BLOG = ROOT / "docs" / "BLOG.md"

# Numbers that are page structure, not claims: they describe the artifact's own
# shape and are verified by the other gates rather than by the papers.
STRUCTURAL = {
    "32", "256", "8", "2", "1", "0", "19", "5",       # shapes, counts, offsets
    "8,192", "27,776", "1024", "13", "17", "3", "4",
    "1e-5", "2.6e-6", "1.9", "100",
}

# Figures the ledger considers load-bearing for the artifact. Each must appear
# on the page AND in CLAIMS.md. Keyed by the CLAIMS.md row that backs it.
LOAD_BEARING = {
    "29.5": "C1 — BDH-CQ pass@2 on public ARC-AGI-1",
    "0.00070": "C1 — computed cost per task",
    "34.2": "C3 — GPT-5.6 Luna (Low), the accuracy comparison",
    "19/24": "C8 — depth-five nesting, short context",
    "24/24": "C8 — depth-five nesting, matched support",
    "13/24": "C9 — length-eight ordering, recovered",
    "0/24": "C9 — length-eight ordering, baseline",
    "72/72": "C10 — rotation composed with relocation",
    "0/72": "C10 — colour swap composed with relocation",
}


def numbers_in(text):
    """Percentages, ratios and fractions a reader would take as a claim."""
    pats = [
        r"\b\d+\.\d+%",          # 29.5%
        r"\b\d+/\d+\b",          # 19/24
        r"\$0\.\d+",             # $0.00070
        r"\b\d+\.\d+×",          # 2.06×
    ]
    found = set()
    for p in pats:
        found |= set(re.findall(p, text))
    return found


def strip_tags(html):
    html = re.sub(r"<script.*?</script>", " ", html, flags=re.S)
    html = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.S)
    return re.sub(r"<[^>]+>", " ", html)


def main():
    for f in (PAGE, CLAIMS):
        if not f.exists():
            sys.exit(f"missing {f}")

    page_text = strip_tags(PAGE.read_text(encoding="utf-8"))
    claims_text = CLAIMS.read_text(encoding="utf-8")
    fails = []

    # ---- direction 1: every literal on the page is in the ledger ----------
    lits = numbers_in(page_text)
    unbacked = []
    for lit in sorted(lits):
        bare = lit.strip("%×$")
        if bare in STRUCTURAL:
            continue
        if lit in claims_text or bare in claims_text:
            continue
        unbacked.append(lit)
    print(f"page literals checked   {len(lits)}")
    if unbacked:
        fails.append(f"{len(unbacked)} page figure(s) absent from CLAIMS.md")
        for u in unbacked:
            print(f"  UNBACKED  {u}  — not found in docs/CLAIMS.md")

    # ---- direction 2: load-bearing ledger figures are on the page --------
    missing = []
    for fig, why in LOAD_BEARING.items():
        on_page = fig in page_text
        in_ledger = fig in claims_text
        if not in_ledger:
            missing.append(f"{fig} missing from CLAIMS.md ({why})")
        if not on_page:
            missing.append(f"{fig} missing from the page ({why})")
    print(f"load-bearing figures    {len(LOAD_BEARING)}")
    if missing:
        fails.append(f"{len(missing)} load-bearing figure problem(s)")
        for m in missing:
            print(f"  MISSING   {m}")

    # ---- direction 3: the docs agree with each other ---------------------
    for doc in (SUMMARY, BLOG):
        if not doc.exists():
            continue
        text = doc.read_text(encoding="utf-8")
        bad = []
        for fig in ("29.5", "34.2", "0/72", "19/24"):
            if fig in text and fig not in claims_text:
                bad.append(fig)
        if bad:
            fails.append(f"{doc.name} cites {bad} not in CLAIMS.md")
            print(f"  MISMATCH  {doc.name}: {bad}")

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        print("\nCLAIMS CHECK FAILED")
        return 1
    print("CLAIMS CHECK PASSED — every page figure is backed by the ledger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
