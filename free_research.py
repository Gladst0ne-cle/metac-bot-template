"""Zero-cost multi-source forecasting research engine using stdlib and requests."""

from __future__ import annotations

import asyncio
import csv
import email.utils
import html
import io
import ipaddress
import json
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

_HTTP_HEADERS = {"User-Agent": "MetaculusResearchBot/1.0 (+https://www.metaculus.com)"}
_BINARY_EXTS = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".zip",
    ".mp4",
    ".mp3",
    ".xlsx",
)
_FRED_ID_RE = re.compile(r"^[A-Z0-9_]{2,25}$")
_FRED_URL_RE = re.compile(r"fred\.stlouisfed\.org/series/([A-Z0-9_]{2,25})", re.I)
_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+")


def _fallback_query(question_text: str, max_tokens: int = 12) -> str:
    tokens = re.sub(r"[^A-Za-z0-9\s\-]", " ", question_text).split()
    return " ".join(tokens[:max_tokens])[:120].strip()


def _is_safe_public_url(url: str) -> bool:
    """Reject internal/private hosts, metaculus.com, and non-HTML binary files."""
    try:
        parsed = urllib.parse.urlparse(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    host = (parsed.hostname or "").lower().strip(".")
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    if host == "metaculus.com" or host.endswith(".metaculus.com"):
        return False
    if any(parsed.path.lower().endswith(ext) for ext in _BINARY_EXTS):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            return False
    except ValueError:
        pass
    return True


def extract_primary_urls(text: str, max_urls: int = 2) -> list[str]:
    """Extract deduplicated safe public URLs from question text or criteria."""
    seen: set[str] = set()
    urls: list[str] = []
    for raw in _URL_RE.findall(text or ""):
        cleaned = raw.rstrip(".,;:)]}>\"'")
        if cleaned not in seen and _is_safe_public_url(cleaned):
            seen.add(cleaned)
            urls.append(cleaned)
            if len(urls) >= max_urls:
                break
    return urls


def fetch_primary_url_excerpt(url: str, max_chars: int = 1400) -> str:
    """Fetch a public resolution/reference webpage and extract clean plain text."""
    if not _is_safe_public_url(url):
        return ""
    try:
        resp = requests.get(url, headers=_HTTP_HEADERS, timeout=10, stream=True)
        resp.raise_for_status()
        if not _is_safe_public_url(resp.url):
            return ""
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if ctype and not any(t in ctype for t in ("text/html", "text/plain", "xml", "json")):
            return ""
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total >= 200_000:
                break
        raw_html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        stripped = re.sub(
            r"(?is)<(script|style|noscript|svg).*?>.*?</\1>", " ", raw_html
        )
        stripped = re.sub(r"(?s)<[^>]+>", " ", stripped)
        clean = re.sub(r"\s+", " ", html.unescape(stripped)).strip()
        return clean[:max_chars] if clean else ""
    except Exception as exc:
        logger.debug(f"Primary URL fetch failed ({url}): {exc}")
        return ""


def _parse_rss_datetime(pub_date: str) -> datetime | None:
    if not pub_date:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(pub_date.strip())
    except Exception:
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_google_news_rss(
    queries: list[str],
    as_of_date: str | None = None,
    max_items: int = 12,
    max_chars: int = 3800,
) -> str:
    """Fetch deduplicated Google News RSS items with optional strict as_of_date filter."""
    collected: list[tuple[float, str, str, str]] = []
    seen_titles: set[str] = set()
    clean_queries = [q.strip()[:120] for q in queries if q and q.strip()][:3]

    for q in clean_queries:
        search_q = f"{q} before:{as_of_date}" if as_of_date else q
        try:
            resp = requests.get(
                "https://news.google.com/rss/search",
                params={"q": search_q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                headers=_HTTP_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logger.debug(f"Google News RSS query failed ({q}): {exc}")
            continue

        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            pub_raw = (item.findtext("pubDate") or "").strip()
            source = (item.findtext("source") or "").strip()
            if not title:
                continue
            dt = _parse_rss_datetime(pub_raw)
            if as_of_date:
                if dt is None or dt.date().isoformat() > as_of_date:
                    continue
            norm = re.sub(r"\W+", " ", title.lower()).strip()
            if not norm or norm in seen_titles:
                continue
            seen_titles.add(norm)
            date_str = dt.date().isoformat() if dt else pub_raw[:16]
            ts = dt.timestamp() if dt else 0.0
            collected.append((ts, date_str, title, source))

    collected.sort(key=lambda row: row[0], reverse=True)
    lines = []
    for _, date_str, title, source in collected[:max_items]:
        src_suffix = f" ({source})" if source else ""
        date_prefix = f"[{date_str}] " if date_str else ""
        lines.append(f"- {date_prefix}{title}{src_suffix}")
    return "\n".join(lines)[:max_chars]


def fetch_fred_series(
    series_ids: list[str],
    as_of_date: str | None = None,
    max_series: int = 2,
) -> str:
    """Fetch St. Louis Fed FRED CSV observations filtered by <= as_of_date."""
    valid_ids: list[str] = []
    for raw_id in series_ids:
        sid = (raw_id or "").strip().upper()
        if sid and _FRED_ID_RE.match(sid) and sid not in valid_ids:
            valid_ids.append(sid)
        if len(valid_ids) >= max_series:
            break

    blocks: list[str] = []
    for sid in valid_ids:
        try:
            resp = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={"id": sid},
                headers=_HTTP_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            reader = csv.reader(io.StringIO(resp.text))
            next(reader, None)  # skip header
            rows: list[tuple[str, float]] = []
            for row in reader:
                if len(row) < 2:
                    continue
                obs_date, raw_val = row[0].strip(), row[1].strip()
                if not obs_date or raw_val in ("", "."):
                    continue
                if as_of_date and obs_date > as_of_date:
                    continue
                try:
                    rows.append((obs_date, float(raw_val)))
                except ValueError:
                    continue
            if not rows:
                continue
            recent = rows[-6:]
            obs_str = ", ".join(f"{d}: {v:g}" for d, v in recent)
            delta_str = ""
            if len(recent) >= 2:
                prev_v, last_v = recent[-2][1], recent[-1][1]
                diff = last_v - prev_v
                if abs(prev_v) > 1e-12:
                    pct = (diff / abs(prev_v)) * 100.0
                    delta_str = f" (latest delta: {diff:+g}, {pct:+.2f}%)"
                else:
                    delta_str = f" (latest delta: {diff:+g})"
            blocks.append(f"- FRED {sid}: {obs_str}{delta_str}")
        except Exception as exc:
            logger.debug(f"FRED fetch failed ({sid}): {exc}")
    return "\n".join(blocks)


def _format_polymarket_hit(m: dict[str, Any]) -> str | None:
    if m.get("closed"):
        return None
    vol = float(m.get("volumeNum") or m.get("volume") or 0)
    if vol < 1000:
        return None
    q_title = (m.get("question") or m.get("groupItemTitle") or "").strip()
    if not q_title:
        return None
    try:
        outcomes = m.get("outcomes")
        prices = m.get("outcomePrices")
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        if isinstance(prices, str):
            prices = json.loads(prices)
        if not outcomes or not prices or len(outcomes) != len(prices):
            return None
        pairs = ", ".join(
            f"{o}={float(p) * 100:.1f}%" for o, p in list(zip(outcomes, prices))[:4]
        )
        return f"- [Polymarket | vol=${vol:,.0f}] {q_title}: {pairs}"
    except Exception:
        return None


def fetch_prediction_markets(query: str, max_chars: int = 1600) -> str:
    """Query Polymarket and Manifold public APIs for live market odds."""
    q = (query or "").strip()[:100]
    if not q:
        return ""
    lines: list[str] = []

    try:
        r_poly = requests.get(
            "https://gamma-api.polymarket.com/public-search",
            params={"q": q, "limit_per_type": 4},
            headers=_HTTP_HEADERS,
            timeout=10,
        )
        r_poly.raise_for_status()
        poly_added = 0
        for ev in r_poly.json().get("events") or []:
            for m in ev.get("markets") or []:
                formatted = _format_polymarket_hit(m)
                if formatted:
                    lines.append(formatted)
                    poly_added += 1
                    if poly_added >= 3:
                        break
            if poly_added >= 3:
                break
    except Exception as exc:
        logger.debug(f"Polymarket search failed ({q}): {exc}")

    try:
        r_man = requests.get(
            "https://api.manifold.markets/v0/search-markets",
            params={"term": q, "filter": "open", "sort": "liquidity", "limit": 3},
            headers=_HTTP_HEADERS,
            timeout=10,
        )
        r_man.raise_for_status()
        for m in r_man.json() or []:
            title = (m.get("question") or "").strip()
            prob = m.get("probability")
            bettors = m.get("uniqueBettorCount") or 0
            if title and prob is not None and bettors >= 5:
                lines.append(
                    f"- [Manifold | {bettors} traders] {title}: Yes={float(prob) * 100:.1f}%"
                )
    except Exception as exc:
        logger.debug(f"Manifold search failed ({q}): {exc}")

    return "\n".join(lines)[:max_chars]


def fetch_wikipedia_summary(query: str, max_chars: int = 900) -> str:
    """Fetch top matching Wikipedia article intro summary."""
    q = (query or "").strip()[:100]
    if not q:
        return ""
    try:
        r_search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": q,
                "srlimit": 1,
                "format": "json",
            },
            headers=_HTTP_HEADERS,
            timeout=8,
        )
        r_search.raise_for_status()
        hits = r_search.json().get("query", {}).get("search") or []
        if not hits:
            return ""
        title = hits[0].get("title")
        if not title:
            return ""
        r_ext = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "titles": title,
                "format": "json",
            },
            headers=_HTTP_HEADERS,
            timeout=8,
        )
        r_ext.raise_for_status()
        pages = r_ext.json().get("query", {}).get("pages") or {}
        for page in pages.values():
            extract = re.sub(r"\s+", " ", (page.get("extract") or "")).strip()
            if extract:
                return f"[{title}] {extract}"[:max_chars]
    except Exception as exc:
        logger.debug(f"Wikipedia summary failed ({q}): {exc}")
    return ""


async def _plan_queries(
    question_text: str,
    resolution_criteria: str,
    planner_llm: Any | None,
) -> dict[str, Any]:
    fb = _fallback_query(question_text)
    default_plan: dict[str, Any] = {
        "news_queries": [fb] if fb else [],
        "market_query": " ".join(fb.split()[:6]),
        "wiki_query": " ".join(fb.split()[:5]),
        "fred_series_ids": [],
    }
    if planner_llm is None:
        return default_plan

    prompt = (
        "Extract crisp English search queries for researching this forecasting question.\n"
        "Return ONLY valid JSON with keys:\n"
        '- "news_queries": list of 2 or 3 short focused keyword search strings (3-7 words each)\n'
        '- "market_query": 2-5 word query for Polymarket/Manifold\n'
        '- "wiki_query": 2-5 word entity/topic for Wikipedia\n'
        '- "fred_series_ids": list of 0-2 exact uppercase FRED series IDs if macroeconomic '
        "(e.g. DFF, CPIAUCSL, UNRATE, GDP), else []\n\n"
        f"Question: {question_text}\n"
        f"Resolution criteria: {resolution_criteria[:600]}"
    )
    try:
        raw = await planner_llm.invoke(prompt)
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return default_plan
        parsed = json.loads(cleaned[start : end + 1])
        news_q = [str(x).strip() for x in (parsed.get("news_queries") or []) if str(x).strip()]
        return {
            "news_queries": news_q[:3] or default_plan["news_queries"],
            "market_query": str(parsed.get("market_query") or default_plan["market_query"]).strip(),
            "wiki_query": str(parsed.get("wiki_query") or default_plan["wiki_query"]).strip(),
            "fred_series_ids": [
                str(x).strip().upper() for x in (parsed.get("fred_series_ids") or [])
            ][:2],
        }
    except Exception as exc:
        logger.debug(f"Query planner fell back to heuristic query: {exc}")
        return default_plan


async def build_free_research_report(
    question_text: str,
    resolution_criteria: str,
    fine_print: str,
    background_info: str,
    planner_llm: Any | None = None,
    as_of_date: str | None = None,
    max_total_chars: int = 9000,
) -> str:
    """Build a multi-source zero-cost research dossier with strict point-in-time safety."""
    plan = await _plan_queries(question_text, resolution_criteria, planner_llm)
    combined_txt = f"{question_text}\n{resolution_criteria}\n{fine_print}\n{background_info}"
    fred_ids = list(plan["fred_series_ids"])
    for m in _FRED_URL_RE.findall(combined_txt):
        if m.upper() not in fred_ids:
            fred_ids.append(m.upper())

    tasks: dict[str, asyncio.Future[str] | Any] = {
        "news": asyncio.to_thread(
            fetch_google_news_rss, plan["news_queries"], as_of_date
        ),
        "fred": asyncio.to_thread(fetch_fred_series, fred_ids, as_of_date),
    }

    # Skip live reference pages, markets, and Wikipedia during historical backtests
    primary_urls: list[str] = []
    if as_of_date is None:
        primary_urls = extract_primary_urls(f"{resolution_criteria}\n{fine_print}", max_urls=2)
        if not primary_urls:
            primary_urls = extract_primary_urls(background_info, max_urls=1)
        for idx, u in enumerate(primary_urls):
            tasks[f"url_{idx}"] = asyncio.to_thread(fetch_primary_url_excerpt, u)
        tasks["markets"] = asyncio.to_thread(
            fetch_prediction_markets, plan["market_query"]
        )
        tasks["wiki"] = asyncio.to_thread(fetch_wikipedia_summary, plan["wiki_query"])

    keys = list(tasks.keys())
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    outputs = {
        k: (r if isinstance(r, str) else "") for k, r in zip(keys, results)
    }

    sections: list[str] = []
    url_blocks = [
        f"Source ({primary_urls[i]}):\n{outputs[f'url_{i}']}"
        for i in range(len(primary_urls))
        if outputs.get(f"url_{i}")
    ]
    if url_blocks:
        sections.append("### Primary Resolution / Reference Pages\n" + "\n\n".join(url_blocks))
    if outputs.get("fred"):
        sections.append("### Official Macro Data (FRED)\n" + outputs["fred"])
    if outputs.get("news"):
        sections.append("### Recent News Headlines (Google News RSS)\n" + outputs["news"])
    if outputs.get("markets"):
        sections.append(
            "### External Prediction Market Benchmarks (Verify Criteria Match)\n"
            + outputs["markets"]
        )
    if outputs.get("wiki"):
        sections.append("### Background Reference (Wikipedia)\n" + outputs["wiki"])

    return "\n\n".join(sections)[:max_total_chars]
