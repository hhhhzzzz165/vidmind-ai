"""video_parser.py — 用 yt-dlp 抓取 B站视频标题与字幕，含防反爬与正则时间轴清洗。

对外接口：
    parse_video(url, cookies_file=None) -> VideoInfo
    clean_transcript(raw) -> str
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

try:
    import yt_dlp
except ImportError:  # pragma: no cover
    yt_dlp = None


class VideoParseError(Exception):
    """视频解析失败（无字幕、需登录、网络错误等）。"""


@dataclass
class VideoInfo:
    title: str
    bvid: str
    url: str
    transcript: str
    duration: Optional[float] = None
    subtitle_lang: Optional[str] = None


# —— 防反爬请求头 ——
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# 字幕语言优先级（含 AI 自动生成）
SUBTITLE_LANGS = ["ai-zh", "zh-CN", "zh-Hans", "zh", "zh-Hant", "zh-TW"]

# —— 正则：时间轴 / 格式噪声 ——
_RE_TIMECODE = re.compile(
    r"^\s*(?:\d{1,3}:)?\d{1,2}:\d{2}[,.]\d{3}\s*-->\s*(?:\d{1,3}:)?\d{1,2}:\d{2}[,.]\d{3}(?:\s+.*)?\s*$"
)
_RE_CUE_INDEX = re.compile(r"^\s*\d{1,4}\s*$")
_RE_TAG = re.compile(r"<[^>]+>")
_RE_VTT_META = re.compile(r"^\s*(WEBVTT|Kind:|Language:|NOTE(?:\s.*)?)$", re.IGNORECASE)
_RE_WATERMARK = re.compile(r"(?:字幕由[^\n]{0,30}?(?:提供|制作|生成|整理)|本字幕由[^\n]{0,40})")
_RE_INLINE_TS = re.compile(r"\d{1,3}:\d{2}:\d{2}[,.]\d{3}|\d{1,2}:\d{2}[,.]\d{3}")
_RE_MULTI_SPACE = re.compile(r"[ \t]+")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_BVID = re.compile(r"(BV[0-9A-Za-z]{10})")


def clean_transcript(raw: str) -> str:
    """用正则清洗字幕：去掉时间轴、cue 序号、VTT 头、标签与水印，返回纯文本。"""
    if not raw:
        return ""
    raw_lines = [ln.strip() for ln in raw.splitlines()]
    out: list[str] = []
    i, n = 0, len(raw_lines)
    while i < n:
        line = raw_lines[i]
        if not line:
            i += 1
            continue
        if _RE_VTT_META.match(line):
            i += 1
            continue
        # cue 序号（如 "1"、"23"）：仅当下一非空行是时间轴时才丢弃，避免误删正文数字行
        if _RE_CUE_INDEX.match(line):
            j = i + 1
            while j < n and not raw_lines[j]:
                j += 1
            if j < n and _RE_TIMECODE.match(raw_lines[j]):
                i += 1
                continue
        if _RE_TIMECODE.match(line):
            i += 1
            continue
        line = _RE_TAG.sub("", line)
        line = _RE_INLINE_TS.sub("", line)
        line = line.strip()
        if line:
            out.append(line)
        i += 1

    text = "\n".join(out)
    text = _RE_WATERMARK.sub("", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    return text.strip()


def parse_video(url: str, cookies_file: Optional[str] = None) -> VideoInfo:
    """抓取 B站视频标题与字幕并清洗。返回 VideoInfo。"""
    if yt_dlp is None:
        raise VideoParseError("缺少依赖 yt-dlp，请先执行 pip install -r requirements.txt")

    info = _extract_info(url, cookies_file)
    title = info.get("title") or "未命名视频"
    bvid = _extract_bvid(url, info)
    duration = info.get("duration")

    track, lang = _pick_subtitle(info)
    if track is None:
        # 回退：让 yt-dlp 直接把字幕写到临时目录再读取
        raw, ext, _ = _download_subs_to_tmp(url, cookies_file)
    else:
        raw, ext = _fetch_subtitle_text(track)

    plain = _subtitle_to_plain(raw, ext)
    transcript = clean_transcript(plain)
    if not transcript:
        raise VideoParseError(
            "字幕解析后为空：该视频可能无字幕，或字幕仅登录可见，请尝试 --cookies cookies.txt"
        )
    return VideoInfo(
        title=title, bvid=bvid, url=url, transcript=transcript,
        duration=duration, subtitle_lang=lang,
    )


def _extract_info(url: str, cookies_file: Optional[str]) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "retries": 3,
        "socket_timeout": 20,
        "http_headers": DEFAULT_HEADERS,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as e:
        raise VideoParseError(f"yt-dlp 解析失败（网络/风控或需 Cookie）：{e}") from e


def _extract_bvid(url: str, info: dict) -> str:
    for c in (url, str(info.get("id") or ""), str(info.get("bvid") or "")):
        m = _RE_BVID.search(c)
        if m:
            return m.group(1)
    return str(info.get("id") or info.get("bvid") or "")


def _pick_subtitle(info: dict):
    """从 info dict 里挑选一条中文字幕轨道。返回 (track, lang) 或 (None, None)。"""
    for src in ("subtitles", "automatic_captions"):
        subs = info.get(src)
        if not isinstance(subs, dict):
            continue
        for lang in SUBTITLE_LANGS:
            for track in subs.get(lang) or []:
                if isinstance(track, dict) and (track.get("url") or track.get("data")):
                    return track, lang
        for lang, tracks in subs.items():
            if "zh" in (lang or "").lower():
                for track in tracks or []:
                    if isinstance(track, dict) and (track.get("url") or track.get("data")):
                        return track, lang
    return None, None


def _fetch_subtitle_text(track: dict) -> tuple[str, str]:
    """优先用 track 内嵌的 data（B站 extractor 已转 SRT），否则带防反爬头抓取 url。"""
    if track.get("data"):
        return str(track["data"]), track.get("ext") or "srt"
    url = track.get("url")
    if not url:
        raise VideoParseError("字幕轨道缺少可用地址")
    req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
    last_err = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            return text, track.get("ext") or _guess_ext(text)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last_err = e
            time.sleep(1.5 * attempt + random.uniform(0, 0.5))
    raise VideoParseError(f"抓取字幕失败（已重试 3 次）：{last_err}")


def _guess_ext(text: str) -> str:
    t = text.lstrip()
    if t.startswith("{"):
        return "json"
    if "WEBVTT" in text[:200]:
        return "vtt"
    return "srt"


def _subtitle_to_plain(raw: str, ext: str) -> str:
    ext = (ext or "").lower()
    if ext == "json" or raw.lstrip().startswith("{"):
        return _json_subtitle_to_text(raw)
    return raw


def _json_subtitle_to_text(raw: str) -> str:
    """B站字幕 JSON 形如 {"body":[{"from":..,"to":..,"content":".."}]}，抽取 content 拼接。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    body = data.get("body") if isinstance(data, dict) else None
    if not isinstance(body, list):
        return raw
    parts = []
    for seg in body:
        if isinstance(seg, dict):
            content = seg.get("content") or seg.get("text") or ""
            if content:
                parts.append(str(content))
        elif isinstance(seg, str):
            parts.append(seg)
    return "\n".join(parts)


def _download_subs_to_tmp(url: str, cookies_file: Optional[str]) -> tuple[str, str, Optional[str]]:
    """回退方案：让 yt-dlp 把字幕写到项目内 _subs_tmp/，读取后清理。"""
    base = os.path.dirname(os.path.abspath(__file__))
    tmpdir = os.path.join(base, "_subs_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": SUBTITLE_LANGS,
        "subtitlesformat": "srt/vtt/best",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "retries": 3,
        "http_headers": DEFAULT_HEADERS,
    }
    if cookies_file:
        opts["cookiefile"] = cookies_file

    content, ext = None, None
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(url, download=True)
        for fn in sorted(os.listdir(tmpdir)):
            path = os.path.join(tmpdir, fn)
            if not os.path.isfile(path):
                continue
            e = fn.rsplit(".", 1)[-1].lower()
            if e in ("srt", "vtt", "json"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content, ext = f.read(), e
                break
    except Exception as e:
        raise VideoParseError(f"yt-dlp 下载字幕失败：{e}") from e
    finally:
        # 沙盒下删除可能被拒，忽略失败；文件很小，留待下次覆盖
        for fn in list(os.listdir(tmpdir)):
            try:
                os.remove(os.path.join(tmpdir, fn))
            except OSError:
                pass

    if content is None:
        raise VideoParseError("未获取到字幕文件（该视频可能无字幕或需登录 Cookie）")
    return content, ext, None
