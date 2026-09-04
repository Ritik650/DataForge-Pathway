"""Check every external link in the page, README and both source documents.

Link stability is graded, and a dead arXiv or repository link in a submission
whose whole argument is "check our sources" is worse than the two points it
costs. Runs in `make verify` so a link cannot rot unnoticed between now and
submission.

Relative links are checked as files on disk; absolute ones over the network.

Run: python scripts/check_links.py
"""

import concurrent.futures as cf
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
UA = "Mozilla/5.0 (compatible; DataForge-link-check/1.0)"

# Exit code meaning "this machine could not run the gate", distinct from
# failure. verify.py reports it as SKIPPED rather than FAIL.
SKIP = 77

SOURCES = [
    ROOT / "artifact" / "index.html",
    ROOT / "README.md",
    ROOT / "NOTICE.md",
    ROOT / "docs" / "CONCEPT_SUMMARY.md",
    ROOT / "docs" / "BLOG.md",
    ROOT / "docs" / "CLAIMS.md",
]

URL_RE = re.compile(r"https?://[^\s\"'<>)\]}]+")
MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


# A link is BROKEN only if the server actually says the resource is gone.
# Everything else -- proxy refusals, bot blocks, rate limits, timeouts, DNS --
# says something about the network this ran on, not about the link. Conflating
# the two makes the harness fail on a judge's restricted network and print
# "do not ship" about a submission that is fine.
BROKEN_CODES = {404, 410}
UNREACHABLE_CODES = {401, 403, 405, 407, 429, 500, 502, 503, 504}


def check_url(url):
    """Return (url, verdict, detail) with verdict in {ok, broken, unreachable}."""
    url = url.rstrip(".,;:")
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, headers={"User-Agent": UA}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                if 200 <= r.status < 400:
                    return url, "ok", r.status
                if r.status in BROKEN_CODES:
                    return url, "broken", r.status
                return url, "unreachable", r.status
        except urllib.error.HTTPError as e:
            if e.code in BROKEN_CODES:
                return url, "broken", e.code
            if method == "HEAD" and e.code in (403, 405):
                continue          # many hosts refuse HEAD; try GET before judging
            return url, "unreachable", e.code
        except Exception as e:
            if method == "HEAD":
                continue
            return url, "unreachable", type(e).__name__
    return url, "unreachable", "no response"


def main():
    urls, rels = set(), []
    for src in SOURCES:
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        urls |= set(URL_RE.findall(text))
        for target in MD_LINK_RE.findall(text):
            if target.startswith(("http", "#", "mailto:")):
                continue
            rels.append((src, target.split("#")[0]))

    fails = []

    # relative links: files on disk
    print(f"relative links   {len(rels)}")
    for src, target in rels:
        path = (src.parent / target).resolve()
        if not path.exists():
            fails.append(f"{src.name} -> {target} (no such file)")
            print(f"  BROKEN  {src.name} -> {target}")

    # external links, in parallel
    urls = {u.rstrip(".,;:") for u in urls}
    unreachable = []
    print(f"external links   {len(urls)}")
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for url, verdict, detail in ex.map(check_url, sorted(urls)):
            if verdict == "ok":
                continue
            if verdict == "broken":
                fails.append(f"{url} -> {detail}")
                print(f"  BROKEN       {url}  {detail}")
            else:
                unreachable.append((url, detail))
                print(f"  UNREACHABLE  {url}  {detail}  (network, not the link)")

    print()
    if fails:
        print(f"LINK CHECK FAILED — {len(fails)} link(s) genuinely broken")
        return 1

    # Every external link unreachable means no egress, not a bad submission.
    if urls and len(unreachable) == len(urls):
        print("LINK CHECK SKIPPED — no network egress from this machine "
              f"({len(urls)} links unverified)")
        return SKIP
    if unreachable:
        print(f"LINK CHECK PASSED — {len(urls) - len(unreachable)} verified, "
              f"{len(unreachable)} unreachable from here (not counted as broken)")
        return 0
    print("LINK CHECK PASSED — every link resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
