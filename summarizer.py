"""summarizer.py — 接入 DeepSeek API（OpenAI 兼容）。

对外接口：
    DeepSeekClient.summarize(title, transcript) -> dict   # 函数A：总结分类
    DeepSeekClient.compare(old_knowledge_base, new_summary) -> dict  # 函数B：对比
    chunk_text(text, max_chars) -> list[str]
    merge_partials(partials, title) -> dict
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


class SummarizeError(Exception):
    """DeepSeek 调用或解析失败。"""


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_CHUNK_SIZE = 6000

_SYSTEM_PROMPT = (
    "你是一名专业的知识整理助手。你会把视频字幕总结成结构化 JSON，并按照知识点分类。"
    "只输出 JSON，不要输出任何解释，也不要输出 Markdown 代码块围栏之外的文字。"
)


def _load_config() -> dict:
    load_dotenv()
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        "model": os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
        "timeout": float(os.getenv("DEEPSEEK_TIMEOUT", "120")),
    }


class DeepSeekClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        cfg = _load_config()
        self.api_key = api_key or cfg["api_key"]
        self.base_url = base_url or cfg["base_url"]
        self.model = model or cfg["model"]
        self.timeout = timeout or cfg["timeout"]
        if not self.api_key:
            raise SummarizeError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置（参考 .env.example）")
        if OpenAI is None:
            raise SummarizeError("缺少依赖 openai，请先执行 pip install -r requirements.txt")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    # —— 函数A：总结分类 ——
    def summarize(self, title: str, transcript: str) -> dict:
        transcript = (transcript or "").strip()
        if not transcript:
            raise SummarizeError("字幕为空，无法总结")
        chunks = chunk_text(transcript)
        if len(chunks) == 1:
            data = self._call_json(_summarize_prompt(title, transcript))
        else:
            partials = [
                self._call_json(_summarize_prompt(title, c, is_chunk=True)) for c in chunks
            ]
            data = merge_partials(partials, title)
        return _normalize_summary(data, title)

    # —— 函数B：对比 ——
    def compare(self, old_knowledge_base: dict, new_summary: dict) -> dict:
        old_points = _collect_old_points(old_knowledge_base)
        data = self._call_json(_compare_prompt(old_points, new_summary))
        return _normalize_compare(data)

    # —— 底层调用 ——
    def _chat(self, messages: list[dict]) -> str:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if "reasoner" not in self.model:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["temperature"] = 0.3
        try:
            resp = self.client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content) or ""
        except Exception as e:
            raise SummarizeError(f"DeepSeek API 调用失败：{e}") from e

    def _call_json(self, prompt: str, retries: int = 2) -> dict:
        last_err: Optional[Exception] = None
        for _ in range(retries + 1):
            try:
                raw = self._chat(
                    [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                )
                return _parse_json(raw)
            except SummarizeError as e:
                last_err = e
        raise SummarizeError(f"DeepSeek 未返回可解析的 JSON（已重试 {retries} 次）：{last_err}")


# —— 长文本分块 ——
def chunk_text(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            for sep in ("\n", "。", "！", "？", "；", "，", " "):
                pos = text.rfind(sep, start, end)
                if pos > start + max_chars // 2:
                    end = pos + 1
                    break
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def merge_partials(partials: list[dict], title: str) -> dict:
    """把多个分块的总结合并：按 topic 去重，合并 key_points/tags。"""
    seen: dict[str, dict] = {}
    order: list[str] = []
    summaries: list[str] = []
    for p in partials:
        if not isinstance(p, dict):
            continue
        if p.get("summary"):
            summaries.append(str(p["summary"]))
        for kp in p.get("knowledge_points") or []:
            if not isinstance(kp, dict):
                continue
            topic = (kp.get("topic") or "").strip()
            if not topic:
                continue
            key = _norm_topic(topic)
            if key not in seen:
                seen[key] = {
                    "topic": topic,
                    "category": kp.get("category") or "",
                    "content": kp.get("content") or "",
                    "key_points": [],
                    "tags": [],
                }
                order.append(key)
            cur = seen[key]
            cur["key_points"] = _dedup_strs(
                cur["key_points"] + [s for s in (kp.get("key_points") or []) if s]
            )
            cur["tags"] = _dedup_strs(cur["tags"] + [s for s in (kp.get("tags") or []) if s])
            if not cur["content"] and kp.get("content"):
                cur["content"] = kp["content"]
            if not cur["category"] and kp.get("category"):
                cur["category"] = kp["category"]
    kps = [seen[k] for k in order]
    summary = "；".join(x for x in summaries if x) or f"《{title}》共整理出 {len(kps)} 个知识点。"
    return {"title": title, "summary": summary, "knowledge_points": kps}


# —— prompt 构造 ——
def _summarize_prompt(title: str, transcript: str, is_chunk: bool = False) -> str:
    schema = {
        "title": "视频标题",
        "summary": "整段视频的整体摘要（2-4 句话）",
        "knowledge_points": [
            {
                "topic": "知识点标题",
                "category": "所属分类（如：基础概念/原理/方法/案例/工具等）",
                "content": "该知识点的内容说明（1-3 句）",
                "key_points": ["关键要点1", "关键要点2"],
                "tags": ["标签1", "标签2"],
            }
        ],
    }
    head = f"请将下面这段视频字幕总结为 JSON。视频标题：{title}。\n"
    if is_chunk:
        head += "这是长字幕的一部分，请只提炼该片段包含的知识点，字段结构与整体一致。\n"
    head += "必须严格按如下 JSON 结构输出（不要省略任何字段）：\n"
    head += json.dumps(schema, ensure_ascii=False, indent=2)
    head += "\n\n字幕内容：\n"
    return head + transcript


def _compare_prompt(old_points: list[dict], new_summary: dict) -> str:
    schema = {
        "overlap": [
            {
                "new_topic": "新视频中的知识点",
                "old_topic": "历史知识点",
                "relation": "相同|相似|包含|部分重叠",
                "description": "重叠说明",
            }
        ],
        "expansion": [
            {"topic": "新增或明显扩充的知识点", "description": "说明", "is_new": True}
        ],
        "analysis": "整体对比分析",
    }
    lines = [
        "请对比「旧知识库」与「新视频总结」，分析知识点的重叠与扩充。",
        "输出严格按如下 JSON 结构（relation 取值只能是：相同/相似/包含/部分重叠）：",
        json.dumps(schema, ensure_ascii=False, indent=2),
        "",
        "【旧知识库知识点】",
    ]
    if old_points:
        for i, p in enumerate(old_points, 1):
            lines.append(f"{i}. {p['topic']}（来自《{p['from_video']}》）：{p['content']}")
    else:
        lines.append("（空：历史知识库尚无任何知识点，此时 overlap 应为空数组，expansion 全部 is_new=true）")
    lines += ["", "【新视频总结】", "标题：" + (new_summary.get("title") or "")]
    lines.append("整体摘要：" + (new_summary.get("summary") or ""))
    lines.append("知识点：")
    for i, kp in enumerate(new_summary.get("knowledge_points") or [], 1):
        lines.append(f"{i}. {kp.get('topic')}（{kp.get('category')}）：{kp.get('content')}")
    return "\n".join(lines)


# —— 归一化 / 工具 ——
def _collect_old_points(kb: dict) -> list[dict]:
    points = []
    for v in (kb or {}).get("videos") or []:
        title = v.get("title") or ""
        for kp in v.get("knowledge_points") or []:
            points.append(
                {
                    "topic": kp.get("topic") or "",
                    "content": kp.get("content") or "",
                    "from_video": title,
                }
            )
    return points


def _normalize_summary(data: dict, title: str) -> dict:
    if not isinstance(data, dict):
        raise SummarizeError("总结结果不是 JSON 对象")
    kps = [
        kp
        for kp in (data.get("knowledge_points") or [])
        if isinstance(kp, dict) and (kp.get("topic") or kp.get("content"))
    ]
    for kp in kps:
        kp.setdefault("category", "")
        kp.setdefault("content", "")
        kp.setdefault("key_points", [])
        kp.setdefault("tags", [])
    return {
        "title": data.get("title") or title,
        "summary": data.get("summary") or "",
        "knowledge_points": kps,
    }


def _normalize_compare(data: dict) -> dict:
    if not isinstance(data, dict):
        raise SummarizeError("对比结果不是 JSON 对象")
    return {
        "overlap": [o for o in (data.get("overlap") or []) if isinstance(o, dict)],
        "expansion": [e for e in (data.get("expansion") or []) if isinstance(e, dict)],
        "analysis": data.get("analysis") or "",
    }


def _parse_json(raw: str) -> dict:
    text = (raw or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        s = text.find(opener)
        if s == -1:
            continue
        e = text.rfind(closer)
        if e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                continue
    raise SummarizeError("模型输出中未找到合法 JSON")


def _norm_topic(t: str) -> str:
    return re.sub(r"\s+", "", (t or "").lower())


def _dedup_strs(items: list) -> list[str]:
    seen, out = set(), []
    for it in items:
        s = str(it).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
