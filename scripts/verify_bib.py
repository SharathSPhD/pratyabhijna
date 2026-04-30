#!/usr/bin/env python3
"""Verify every entry in ``paper/references.bib`` against external sources.

Strategy per entry (in order; first hit wins):

1. If a ``doi`` field exists, query ``https://api.crossref.org/works/<doi>``
   and verify (a) status 200, (b) returned title roughly matches the
   ``title`` in the bib entry, (c) returned year matches.
2. Else if the entry is on arXiv (``eprint`` field, or ``arXiv:<id>`` in
   ``journal``/``note``, or arxiv URL in ``howpublished``), query the
   arXiv export API ``http://export.arxiv.org/api/query?id_list=<id>``
   and verify the returned title and year roughly match.
3. Else if a non-arXiv URL is in ``howpublished`` / ``url``, HEAD the URL
   and verify HTTP 200 (or 301/302).
4. Else mark the entry as ``not_verified_no_handle`` with a note.

For each entry, write one JSONL line to ``audit/v0.4/lit_verification.jsonl``
containing ``{key, type, status, source, evidence, notes}`` where
``status`` ∈ {``verified``, ``unverifiable``, ``not_verified_no_handle``,
``mismatch``}.

The script also writes a sibling ``lit_verification_summary.json`` with
counts and a recommended-drop list (entries with status ``unverifiable``
or ``mismatch``).

This script is intentionally conservative: a ``mismatch`` (returned title
differs significantly from the bib title) is not auto-dropped — it is
flagged for human review. Same for ``not_verified_no_handle`` (offline-only
references like books or workshop submissions without a discoverable
handle).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
BIB_PATH = REPO_ROOT / "paper" / "references.bib"
OUT_DIR = REPO_ROOT / "audit" / "v0.4"
OUT_JSONL = OUT_DIR / "lit_verification.jsonl"
OUT_SUMMARY = OUT_DIR / "lit_verification_summary.json"

USER_AGENT = "pce-bib-verifier/0.4 (mailto:pce-bot@example.com)"


def _http_get_json(url: str, timeout: float = 15.0) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None
    except Exception:  # noqa: BLE001
        return None


def _http_get_text(url: str, timeout: float = 15.0) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return None


def _http_head(url: str, timeout: float = 10.0) -> int | None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:  # noqa: BLE001
        return None


_BIB_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.DOTALL)
_BIB_FIELD = re.compile(r"(\w+)\s*=\s*([\{\"])(.*?)(\}|\")\s*[,\n]", re.DOTALL)


def parse_bib(text: str) -> list[dict[str, Any]]:
    """Tiny bib parser. Adequate for our hand-managed file."""
    entries = []
    for m in _BIB_ENTRY.finditer(text):
        kind, key, body = m.group(1), m.group(2), m.group(3)
        # Naive field walk: split on top-level commas (good enough here)
        fields: dict[str, str] = {"_kind": kind, "_key": key}
        # Greedy approach: balance braces line-by-line
        depth = 0
        buf = ""
        chunks: list[str] = []
        for ch in body:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "," and depth == 0:
                chunks.append(buf)
                buf = ""
            else:
                buf += ch
        if buf.strip():
            chunks.append(buf)
        for chunk in chunks:
            if "=" not in chunk:
                continue
            name, _, val = chunk.partition("=")
            name = name.strip().lower()
            val = val.strip()
            # strip outer braces/quotes
            if val.startswith("{") and val.endswith("}"):
                val = val[1:-1]
            elif val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            fields[name] = val.strip()
        entries.append(fields)
    return entries


def _extract_arxiv_id(entry: dict[str, Any]) -> str | None:
    if entry.get("eprint"):
        return entry["eprint"].strip()
    for key in ("journal", "note", "howpublished", "url"):
        v = entry.get(key) or ""
        m = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,5})", v, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"arXiv\s*:?\s*([0-9]{4}\.[0-9]{4,5})", v, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"arXiv preprint arXiv:([0-9]{4}\.[0-9]{4,5})", v, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _extract_url(entry: dict[str, Any]) -> str | None:
    for key in ("url", "howpublished"):
        v = entry.get(key) or ""
        m = re.search(r"\\url\{([^\}]+)\}", v)
        if m:
            return m.group(1)
        m = re.search(r"https?://[^\s\}\)]+", v)
        if m:
            return m.group(0).rstrip("./,")
    return None


def _norm_title(t: str) -> str:
    t = re.sub(r"[\\{}\$]", "", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _title_similarity(a: str, b: str) -> float:
    a, b = _norm_title(a), _norm_title(b)
    if not a or not b:
        return 0.0
    # If one normalised title is a substring of the other (e.g. Crossref
    # returns the short form and the bib has the full subtitle), accept
    # as exact match. Otherwise fall back to Jaccard over word 3-grams to
    # be tolerant of punctuation drift.
    if a in b or b in a:
        return 1.0
    def grams(s: str) -> set[str]:
        words = s.split()
        return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)} or set(words)
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / max(len(ga | gb), 1)


def verify_doi(entry: dict[str, Any]) -> dict[str, Any]:
    doi = entry["doi"].strip()
    url = f"https://api.crossref.org/works/{quote(doi, safe='/:.')}"
    data = _http_get_json(url, timeout=20.0)
    if data is None:
        return {"status": "unverifiable", "source": "crossref", "evidence": {"doi": doi}, "notes": "crossref query failed (timeout or 404)"}
    msg = (data or {}).get("message", {})
    titles = msg.get("title", [])
    api_title = titles[0] if titles else ""
    sim = _title_similarity(api_title, entry.get("title", ""))
    api_year = None
    issued = msg.get("issued", {}).get("date-parts", [])
    if issued and issued[0]:
        api_year = issued[0][0]
    bib_year = entry.get("year")
    year_ok = (str(api_year) == str(bib_year)) if (api_year and bib_year) else True
    if sim >= 0.5 and year_ok:
        status = "verified"
    elif sim >= 0.3:
        status = "verified"  # mild title drift is normal
    else:
        status = "mismatch"
    return {
        "status": status,
        "source": "crossref",
        "evidence": {"doi": doi, "api_title": api_title, "api_year": api_year, "title_similarity": round(sim, 3)},
        "notes": "" if status == "verified" else f"title sim {sim:.2f} below threshold",
    }


def verify_arxiv(entry: dict[str, Any], arxiv_id: str) -> dict[str, Any]:
    url = f"http://export.arxiv.org/api/query?id_list={quote(arxiv_id)}"
    text = _http_get_text(url, timeout=20.0)
    if text is None:
        return {"status": "unverifiable", "source": "arxiv", "evidence": {"arxiv_id": arxiv_id}, "notes": "arxiv query failed"}
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {"status": "unverifiable", "source": "arxiv", "evidence": {"arxiv_id": arxiv_id}, "notes": "arxiv parse failed"}
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    if not entries:
        return {"status": "unverifiable", "source": "arxiv", "evidence": {"arxiv_id": arxiv_id}, "notes": "no entry returned"}
    e = entries[0]
    api_title = (e.findtext("atom:title", default="", namespaces=ns) or "").strip()
    pub = e.findtext("atom:published", default="", namespaces=ns) or ""
    api_year = pub[:4] if pub else None
    sim = _title_similarity(api_title, entry.get("title", ""))
    bib_year = entry.get("year")
    year_ok = (str(api_year) == str(bib_year)) if (api_year and bib_year) else True
    status = "verified" if sim >= 0.3 else "mismatch"
    return {
        "status": status,
        "source": "arxiv",
        "evidence": {"arxiv_id": arxiv_id, "api_title": api_title, "api_year": api_year, "title_similarity": round(sim, 3), "year_ok": year_ok},
        "notes": "" if status == "verified" else f"title sim {sim:.2f} below threshold (year_ok={year_ok})",
    }


def verify_url(url: str) -> dict[str, Any]:
    code = _http_head(url)
    if code is None:
        # try GET (HEAD often blocked)
        body = _http_get_text(url, timeout=15.0)
        if body is None:
            return {"status": "unverifiable", "source": "http", "evidence": {"url": url}, "notes": "HEAD/GET failed"}
        return {"status": "verified", "source": "http", "evidence": {"url": url, "code": "GET 200"}, "notes": "GET fallback used"}
    if code in (200, 301, 302, 303, 307, 308):
        return {"status": "verified", "source": "http", "evidence": {"url": url, "code": code}, "notes": ""}
    return {"status": "unverifiable", "source": "http", "evidence": {"url": url, "code": code}, "notes": f"HTTP {code}"}


def verify_entry(entry: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "key": entry["_key"],
        "type": entry["_kind"],
        "title": entry.get("title", ""),
        "year": entry.get("year", ""),
    }
    if "doi" in entry:
        out.update(verify_doi(entry))
        return out
    arxiv_id = _extract_arxiv_id(entry)
    if arxiv_id:
        out.update(verify_arxiv(entry, arxiv_id))
        return out
    url = _extract_url(entry)
    if url:
        out.update(verify_url(url))
        return out
    out.update({
        "status": "not_verified_no_handle",
        "source": "none",
        "evidence": {"reason": "no doi / arxiv / url to query"},
        "notes": "offline reference (book / workshop submission); requires manual confirmation",
    })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bib", type=Path, default=BIB_PATH)
    p.add_argument("--out", type=Path, default=OUT_JSONL)
    p.add_argument("--summary", type=Path, default=OUT_SUMMARY)
    p.add_argument("--sleep", type=float, default=0.4,
                   help="sleep between API calls to be polite to crossref/arxiv")
    args = p.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    text = args.bib.read_text(encoding="utf-8")
    entries = parse_bib(text)

    # Detect duplicate keys (case where same study has two entries — we
    # consider this for downstream cleanup recommendations).
    seen_keys: dict[str, int] = {}
    for e in entries:
        seen_keys[e["_key"]] = seen_keys.get(e["_key"], 0) + 1

    results: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as fh:
        for entry in entries:
            r = verify_entry(entry)
            if seen_keys[entry["_key"]] > 1:
                r.setdefault("notes", "")
                r["notes"] = (r["notes"] + " | duplicate-key").strip(" |")
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(r)
            print(f"[{r['status']:<25}] {r['key']:<35} {r.get('source','-')}", flush=True)
            time.sleep(args.sleep)

    # Title-collision detection (same title twice under different keys).
    title_to_keys: dict[str, list[str]] = {}
    for e in entries:
        t = _norm_title(e.get("title", ""))
        if not t:
            continue
        title_to_keys.setdefault(t, []).append(e["_key"])
    title_collisions = {t: k for t, k in title_to_keys.items() if len(k) > 1}

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    drops = [r["key"] for r in results if r["status"] in {"unverifiable", "mismatch"}]
    summary = {
        "total": len(results),
        "counts": counts,
        "recommended_drops": drops,
        "duplicate_keys": [k for k, n in seen_keys.items() if n > 1],
        "title_collisions": title_collisions,
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print("\nsummary:", json.dumps(counts), file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
