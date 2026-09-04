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


def check_url(url):
    url = url.rstrip(".,;:")
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return url, r.status, None
    except urllib.error.HTTPError as e:
        if e.code in (403, 405):  # some hosts refuse HEAD; retry with GET
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=25) as r:
                    return url, r.status, None
            except Exception as e2:
                return url, None, str(e2)
        return url, e.code, str(e)
    except Exception as e:
        return url, None, str(e)


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
    print(f"external links   {len(urls)}")
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for url, status, err in ex.map(check_url, sorted(urls)):
            if status and 200 <= status < 400:
                continue
            fails.append(f"{url} -> {status or err}")
            print(f"  BROKEN  {url}  {status or err}")

    print()
    if fails:
        print(f"LINK CHECK FAILED — {len(fails)} broken")
        return 1
    print("LINK CHECK PASSED — every link resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
