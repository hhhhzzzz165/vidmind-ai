"""app.py — VidMind AI Streamlit 前端。

正常模式：真实解析 B站 视频 + DeepSeek 分析 + 导出 Word。
Mock 模式：勾选后使用内置示例（不联网、不调 API），与真实模式完全分开。
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime

import streamlit as st

from main import (
    _mock_compare,
    _mock_summary,
    _safe_filename,
    _upsert_video,
    load_knowledge_base,
    save_knowledge_base,
)
from summarizer import DeepSeekClient, SummarizeError
from video_parser import VideoParseError, parse_video
from word_exporter import export_report

BASE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(BASE, "knowledge_base.json")
MOCK_TITLE = "示例视频（mock）"

_BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")


def _writable_dir(preferred: str, name: str) -> str:
    """优先用项目内 output/；只读环境（如云平台只读挂载）回退到临时目录。"""
    try:
        os.makedirs(preferred, exist_ok=True)
        probe = os.path.join(preferred, ".w_probe")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return preferred
    except OSError:
        d = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(d, exist_ok=True)
        return d


OUT_DIR = _writable_dir(os.path.join(BASE, "output"), "vidmind_output")


def _is_bilibili(u: str) -> bool:
    return "bilibili.com" in u or bool(_BV_RE.search(u))


def _video_error(e: VideoParseError) -> str:
    s = str(e)
    low = s.lower()
    if "仅登录" in s or "需登录" in s or "登录可见" in s or "logged in" in low:
        return f"视频需要登录或 Cookie 无效，请上传有效的 cookies.txt。详情：{s}"
    if "字幕" in s:
        return f"视频没有可用字幕。详情：{s}"
    return f"视频解析失败：{s}"


def _get_api_key():
    """云端用 st.secrets；本地回退到环境变量 / .env。"""
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("DEEPSEEK_API_KEY")


def _normalize_netscape(text: str) -> str:
    """把上传内容整理成 yt-dlp 能解析的标准 Netscape Cookie 文件。

    常见问题修复：缺少标准头、path 列为空、列数不足。
    """
    rows = []
    for line in text.splitlines():
        raw = line.rstrip("\r\n")
        if not raw.strip():
            continue
        prefix = ""
        body = raw
        if body.startswith("#HttpOnly_"):
            prefix, body = "#HttpOnly_", body[len("#HttpOnly_"):]
        elif raw.startswith("#"):
            continue  # 丢弃注释/旧头部，统一重加标准头
        parts = body.split("\t", 6)
        if len(parts) < 6:
            continue
        if len(parts) == 6:
            domain, sub, secure, expiry, name, value = parts
            path = "/"
        else:
            domain, sub, path, secure, expiry, name, value = parts
        domain = domain.strip()
        name = name.replace("\t", " ").strip()
        value = value.replace("\t", " ").strip()
        if not domain or not name:
            continue
        path = path.strip() or "/"
        sub = sub.strip().upper()
        if sub not in ("TRUE", "FALSE"):
            sub = "TRUE" if domain.startswith(".") else "FALSE"
        secure = secure.strip().upper()
        if secure not in ("TRUE", "FALSE"):
            secure = "FALSE"
        expiry = expiry.strip() or "0"
        rows.append(prefix + "\t".join([domain, sub, path, secure, expiry, name, value]))
    return "# Netscape HTTP Cookie File\n" + "\n".join(rows) + "\n"


def _write_temp_cookies(uploaded):
    """把上传的 cookies.txt（先做格式修复）临时写入 output/，调用方负责删除。"""
    if uploaded is None:
        return None
    text = _normalize_netscape(uploaded.getvalue().decode("utf-8", errors="replace"))
    os.makedirs(OUT_DIR, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="vm_cookies_", suffix=".txt", dir=OUT_DIR)
    os.close(fd)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def _remove_quietly(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _unique_docx_path(title: str) -> str:
    """标题 + 时间戳命名；同名再追加序号，保证不覆盖。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{_safe_filename(title)[:60]}_{ts}"
    path = os.path.join(OUT_DIR, stem + ".docx")
    i = 1
    while os.path.exists(path):
        path = os.path.join(OUT_DIR, f"{stem}_{i}.docx")
        i += 1
    return path


def _export(summary, comparison, title, video_url, bvid) -> str:
    docx_path = _unique_docx_path(title)
    meta = {
        "url": video_url,
        "bvid": bvid,
        "title": title,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    export_report(summary, comparison, meta, docx_path)
    return docx_path


def _kb_without(kb, bvid):
    """对比时排除当前视频自身的历史条目，避免「和自己比」。"""
    return {"videos": [v for v in (kb or {}).get("videos", []) if v.get("bvid") != bvid]}


st.set_page_config(page_title="VidMind AI", page_icon="🎬")
st.title("🎬 VidMind AI")
st.caption("B站视频总结 · 知识点分类 · 导出 Word")

url = st.text_input("B站视频链接", placeholder="https://www.bilibili.com/video/BV...")
cookies = st.file_uploader("上传 cookies.txt（可选，仅需登录才能获取的字幕才用）", type=["txt"])
with st.expander("❓ 如何导出自己的 cookies.txt（点开看步骤）"):
    st.markdown(
        """
**为什么需要**：B站 的 AI 字幕接口要求登录态，匿名访问会提示「仅登录可见」。Cookie 只在你本机临时读取、解析完立即删除，不会写进 Word。

**用 Cookie-Editor 导出**
1. 浏览器打开 `https://www.bilibili.com`，**确认已登录**。
2. 点右上角 Cookie-Editor 图标。
3. 面板底部点「导出」图标（向下箭头 ⤓）。
4. 格式选 **Netscape**（不要选 JSON / Header）。
5. 有 Download 就下载；只有 Copy 就粘贴到记事本，保存为 `cookies.txt`（UTF-8）。

**自检**：文件开头应有 `# Netscape HTTP Cookie File`，且必须包含 `SESSDATA` 这一行（没有 = 未登录）。

**隐私提醒**：`cookies.txt` 等同账号登录凭证，**不要发给别人**；本页面只在本机临时使用，用完即删。
"""
    )
mock_mode = st.checkbox("离线 Mock 模式（不联网、不调 API）", value=False)

if st.button("生成总结", type="primary", disabled=not url.strip()):
    st.session_state.pop("result", None)
    url = url.strip()

    if mock_mode:
        summary = _mock_summary(MOCK_TITLE)
        comparison = _mock_compare(load_knowledge_base(KB_PATH), summary)
        try:
            docx = _export(summary, comparison, MOCK_TITLE, url, "BV_mock")
        except Exception as e:
            st.error(f"Word 文件生成失败：{e}")
            docx = None
        st.session_state["result"] = {
            "summary": summary, "comparison": comparison, "docx": docx,
            "title": MOCK_TITLE, "url": url, "bvid": "BV_mock",
        }
    else:
        if not _is_bilibili(url):
            st.error("视频链接无效：请输入 B站 视频链接（如 https://www.bilibili.com/video/BV...）")
        else:
            cookie_path = None
            try:
                cookie_path = _write_temp_cookies(cookies)
                if cookies is not None and "SESSDATA" not in cookies.getvalue().decode("utf-8", errors="replace"):
                    st.warning(
                        "cookies.txt 中没有 SESSDATA（B站 登录态），可能无法获取需登录的字幕；"
                        "请登录 B站 后用「Get cookies.txt LOCALLY / Cookie-Editor」导出完整 Cookie。"
                    )
                with st.spinner("解析视频与字幕..."):
                    info = parse_video(url, cookies_file=cookie_path)
                with st.spinner("DeepSeek 正在总结与分类..."):
                    client = DeepSeekClient(api_key=_get_api_key())
                    summary = client.summarize(info.title, info.transcript)
                with st.spinner("对比历史知识库..."):
                    old_kb = load_knowledge_base(KB_PATH)
                    comparison = client.compare(_kb_without(old_kb, info.bvid), summary)
                try:
                    _upsert_video(old_kb, {
                        "bvid": info.bvid,
                        "title": info.title,
                        "url": info.url,
                        "summarized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "summary": summary.get("summary") or "",
                        "knowledge_points": summary.get("knowledge_points") or [],
                    })
                    save_knowledge_base(KB_PATH, old_kb)
                except Exception as e:
                    st.warning(f"知识库保存失败：{e}")
                try:
                    docx = _export(summary, comparison, info.title, info.url, info.bvid)
                except Exception as e:
                    st.error(f"Word 文件生成失败：{e}")
                    docx = None
                st.session_state["result"] = {
                    "summary": summary, "comparison": comparison, "docx": docx,
                    "title": info.title, "url": info.url, "bvid": info.bvid,
                }
            except VideoParseError as e:
                st.error(_video_error(e))
            except SummarizeError as e:
                st.error(f"DeepSeek API 调用失败：{e}")
            except Exception as e:
                st.error(f"处理失败：{type(e).__name__}: {e}")
            finally:
                _remove_quietly(cookie_path)

res = st.session_state.get("result")
if res:
    s = res["summary"]
    st.subheader("视频总结")
    st.caption(f"标题：{res['title']} ｜ BVID：{res['bvid']} ｜ 链接：{res['url']}")
    st.write(s["summary"])

    st.subheader("知识点分类")
    for kp in s["knowledge_points"]:
        with st.expander(kp["topic"]):
            st.write(f"**分类**：{kp['category']}")
            st.write(kp["content"])
            for p in kp["key_points"]:
                st.markdown(f"- {p}")

    comp = res["comparison"]
    st.subheader("与历史知识库对比")
    if comp.get("overlap"):
        st.dataframe(
            [
                {
                    "新视频知识点": o.get("new_topic", ""),
                    "历史知识点": o.get("old_topic", ""),
                    "关系": o.get("relation", ""),
                    "说明": o.get("description", ""),
                }
                for o in comp["overlap"]
            ]
        )
    else:
        st.write("（无重叠知识点）")

    if comp.get("expansion"):
        st.markdown("**新增 / 扩充知识点：**")
        for e in comp["expansion"]:
            st.markdown(f"- **{e.get('topic')}**：{e.get('description') or ''}")

    if comp.get("analysis"):
        st.info(comp["analysis"])

    if res.get("docx") and os.path.exists(res["docx"]):
        with open(res["docx"], "rb") as f:
            st.download_button(
                "⬇️ 下载 Word",
                f.read(),
                file_name=os.path.basename(res["docx"]),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
