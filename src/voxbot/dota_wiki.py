"""Scrape voice line samples from the Dota 2 Fandom wiki for voice training."""

import asyncio
import pathlib
import random
import re
import shutil
import subprocess
import tempfile
import time
import urllib.parse

import bs4
import niquests
import structlog

_LOGGER = structlog.get_logger(__name__)

WIKI_BASE = "https://dota2.fandom.com"
RESPONSES_CATEGORY_PATH = "/wiki/Category:Responses"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

TARGET_MIN_SECONDS = 20.0
TARGET_MAX_SECONDS = 30.0
CANDIDATE_POOL_SIZE = 40
EXPORT_FORMAT = "mp3"
EXPORT_BITRATE = "128k"

_CF_CHALLENGE_MARKER = "Just a moment..."
_FETCH_RETRIES = 4


class WikiError(Exception):
    """Raised when scraping fails in an expected way (hero missing, no clips, etc.)."""


def _session() -> niquests.Session:
    s = niquests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
        "Referer": "https://dota2.fandom.com/",
    })
    return s


def _fetch(session: niquests.Session, url: str, must_contain: str | None = None) -> str:
    """GET with retries that survive Cloudflare's transient interstitial challenge."""
    last_err: Exception | None = None
    for attempt in range(_FETCH_RETRIES):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            text = resp.text
            if text is None:
                raise WikiError("empty response body")
            if _CF_CHALLENGE_MARKER in text[:2000]:
                last_err = WikiError("cloudflare challenge")
            elif must_contain is not None and must_contain not in text:
                last_err = WikiError(f"expected marker missing: {must_contain}")
            else:
                return text
        except Exception as exc:
            last_err = exc
        time.sleep(0.5 * (attempt + 1))
    raise WikiError(f"failed to fetch {url}: {last_err}")


def list_heroes(session: niquests.Session) -> dict[str, str]:
    """List all heroes found in the Category:Responses wiki page."""
    heroes: dict[str, str] = {}
    api_url = f"{WIKI_BASE}/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": "Category:Responses",
        "cmlimit": "500",
        "format": "json",
        "formatversion": "2",
    }
    while True:
        data = None
        last_err = None
        for attempt in range(_FETCH_RETRIES):
            try:
                resp = session.get(api_url, params=params, timeout=20)
                _LOGGER.info("api_response", url=resp.url, status=resp.status_code, content_type=resp.headers.get("content-type", ""), preview=(resp.text or "")[:500])
                resp.raise_for_status()
                try:
                    data = resp.json()
                except ValueError as exc:
                    last_err = f"invalid JSON: {exc}, preview: {(resp.text or "")[:500]}"
                    _LOGGER.warning("api_json_fail", hero="list_heroes", attempt=attempt, status=resp.status_code, preview=(resp.text or "")[:500])
                    time.sleep(1 * (attempt + 1))
                    continue
                if isinstance(data, dict) and "error" in data:
                    last_err = f"API error: {data['error'].get('info', 'unknown')}"
                    _LOGGER.warning("api_error", error=data["error"])
                    time.sleep(1 * (attempt + 1))
                    continue
                break
            except Exception as exc:
                last_err = str(exc)
                _LOGGER.warning("api_request_fail", attempt=attempt, error=str(exc))
                time.sleep(1 * (attempt + 1))
        if data is None:
            raise WikiError(f"failed to list heroes: {last_err}")
        for page in (data.get("query") or {}).get("categorymembers") or []:
            title = page["title"]
            if title.endswith("/Responses"):
                slug = title.replace("/Responses", "")
                name = urllib.parse.unquote(slug).replace("_", " ")
                heroes[name.lower()] = name
        if "continue" not in data:
            break
        params["cmcontinue"] = data["continue"]["cmcontinue"]
    return heroes


def find_hero(session: niquests.Session, query: str) -> str | None:
    return list_heroes(session).get(query.lower())


def get_audio_urls(session: niquests.Session, hero: str) -> list[str]:
    """Collect original recording audio URLs for a hero's response lines.

    The Dota wiki lays out variant voice lines in side-by-side columns within a
    single list item: the first column is the original recording, subsequent
    columns are augmented (Aghanim's Shard, persona, item-equipped, etc.).
    Pulling the first audio per list item yields the original recordings only.
    """
    slug = urllib.parse.quote(hero.replace(" ", "_"))
    api_url = f"{WIKI_BASE}/api.php"
    params = {
        "action": "parse",
        "page": f"{slug}/Responses",
        "prop": "text",
        "format": "json",
        "formatversion": "2",
    }
    data = None
    last_err = None
    for attempt in range(_FETCH_RETRIES):
        try:
            resp = session.get(api_url, params=params, timeout=20)
            _LOGGER.info("api_response", hero=hero, url=resp.url, status=resp.status_code, content_type=resp.headers.get("content-type", ""), preview=(resp.text or "")[:500])
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError as exc:
                last_err = f"invalid JSON: {exc}, preview: {(resp.text or "")[:500]}"
                _LOGGER.warning("api_json_fail", hero=hero, attempt=attempt, status=resp.status_code, preview=(resp.text or "")[:500])
                time.sleep(1 * (attempt + 1))
                continue
            if isinstance(data, dict) and "error" in data:
                last_err = f"API error: {data['error'].get('info', 'unknown')}"
                _LOGGER.warning("api_error", hero=hero, error=data["error"])
                time.sleep(1 * (attempt + 1))
                continue
            break
        except Exception as exc:
            last_err = str(exc)
            _LOGGER.warning("api_request_fail", hero=hero, attempt=attempt, error=str(exc))
            time.sleep(1 * (attempt + 1))
    if data is None:
        raise WikiError(f"failed to get audio urls for {hero}: {last_err}")
    html = (data.get("parse") or {}).get("text", "")
    if not html:
        raise WikiError(f"no page content for {hero}")

    soup = bs4.BeautifulSoup(html, "html.parser")
    seen: set[str] = set()
    out: list[str] = []
    for li in soup.find_all("li"):
        source = li.find("source")
        if source is None:
            continue
        url = source.get("src")
        if url is None or not isinstance(url, str) or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _ffprobe_duration(path: pathlib.Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def _ffmpeg_concat(inputs: list[pathlib.Path], output: pathlib.Path) -> None:
    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    streams = "".join(f"[{i}:a]" for i in range(len(inputs)))
    cmd += [
        "-filter_complex", f"{streams}concat=n={len(inputs)}:v=0:a=1[out]",
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", EXPORT_BITRATE,
        str(output),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _sample(hero_query: str) -> tuple[str, bytes, str]:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise WikiError("ffmpeg/ffprobe not found on PATH")

    session = _session()
    hero = find_hero(session, hero_query)
    if hero is None:
        raise WikiError(f"hero not found in Category:Responses: {hero_query!r}")

    urls = get_audio_urls(session, hero)
    if not urls:
        raise WikiError(f"no voice line audio found for {hero}")

    random.shuffle(urls)
    max_s = TARGET_MAX_SECONDS
    min_s = TARGET_MIN_SECONDS

    with tempfile.TemporaryDirectory(prefix="voxbot_wiki_") as td:
        tmp = pathlib.Path(td)
        candidates: list[tuple[pathlib.Path, float]] = []
        for i, url in enumerate(urls):
            if len(candidates) >= CANDIDATE_POOL_SIZE:
                break
            ext = pathlib.Path(urllib.parse.urlparse(url).path).suffix.lower() or ".ogg"
            local = tmp / f"clip_{i:03d}{ext}"
            try:
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
                if resp.content is None:
                    raise WikiError("empty response content")
                local.write_bytes(resp.content)
                dur = _ffprobe_duration(local)
            except Exception as exc:
                _LOGGER.warning("wiki_clip_skipped", url=url, error=str(exc))
                continue
            if dur > max_s:
                continue
            candidates.append((local, dur))

        if not candidates:
            raise WikiError(f"no usable voice line clips for {hero}")

        candidates.sort(key=lambda x: x[1], reverse=True)

        chosen: list[pathlib.Path] = []
        total = 0.0
        for path, dur in candidates:
            if total + dur > max_s:
                continue
            chosen.append(path)
            total += dur
            if total >= min_s:
                break

        if not chosen:
            raise WikiError(f"no usable voice line clips for {hero}")

        out_path = tmp / f"out.{EXPORT_FORMAT}"
        _ffmpeg_concat(chosen, out_path)
        data = out_path.read_bytes()

    safe = re.sub(r"[^A-Za-z0-9]+", "_", hero).strip("_") or "hero"
    filename = f"{safe}.{EXPORT_FORMAT}"
    _LOGGER.info("wiki_sample_ready", hero=hero, clips=len(chosen), seconds=round(total, 2), bytes=len(data))
    return hero, data, filename


async def sample_voice_lines(hero_query: str) -> tuple[str, bytes, str]:
    """Async wrapper around the blocking sample function."""
    return await asyncio.to_thread(_sample, hero_query)
