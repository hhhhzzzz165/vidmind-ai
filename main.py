"""main.py — 测试入口。

串联：解析视频 -> AI总结 -> 读历史JSON -> AI对比 -> 导出Word -> 保存JSON。
用法：
    python main.py <B站视频URL>
    python main.py <占位> --mock          # 离线测试，不联网不耗 API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

from dotenv import load_dotenv

from summarizer import DeepSeekClient, SummarizeError
from video_parser import VideoParseError, parse_video
from word_exporter import export_report

DEFAULT_KB = "knowledge_base.json"
DEFAULT_OUT_DIR = "output"
SAMPLE_TRANSCRIPT = os.path.join("sample", "sample_transcript.txt")


def load_knowledge_base(path: str = DEFAULT_KB) -> dict:
    if not os.path.exists(path):
        return {"videos": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "videos" not in data:
        raise ValueError(f"知识库文件 {path} 结构不合法（应含 videos 列表），请检查或备份后处理")
    return data


def save_knowledge_base(path: str, kb: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _upsert_video(kb: dict, entry: dict) -> None:
    bvid = entry.get("bvid")
    videos = kb.setdefault("videos", [])
    for i, v in enumerate(videos):
        if bvid and v.get("bvid") == bvid:
            videos[i] = entry
            return
    videos.append(entry)


def _safe_filename(title: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n]+', "_", title or "video").strip()
    s = re.sub(r"\s+", "_", s)
    return s[:80] or "video"


def _mock_summary(title: str) -> dict:
    return {
        "title": title,
        "summary": f"这是对《{title}》的 mock 总结，用于离线验证流水线。",
        "knowledge_points": [
            {
                "topic": "过拟合",
                "category": "基础概念",
                "content": "模型在训练集上表现好、测试集上表现差的现象。",
                "key_points": ["训练集/测试集表现差距", "常见原因：数据少、模型复杂、噪声多"],
                "tags": ["机器学习", "泛化"],
            },
            {
                "topic": "正则化（L1/L2）",
                "category": "方法",
                "content": "通过在损失函数中加入权重惩罚项来抑制过拟合。",
                "key_points": ["L2 即权重衰减", "L1 使权重稀疏、可做特征选择"],
                "tags": ["正则化", "特征选择"],
            },
            {
                "topic": "交叉验证",
                "category": "方法",
                "content": "把数据分成若干份轮流做验证集，评估模型泛化能力。",
                "key_points": ["K 折划分", "用于评估泛化能力"],
                "tags": ["模型评估"],
            },
        ],
    }


def _mock_compare(old_kb: dict, new_summary: dict) -> dict:
    has_history = bool((old_kb or {}).get("videos"))
    if has_history:
        return {
            "overlap": [
                {
                    "new_topic": "过拟合",
                    "old_topic": "过拟合",
                    "relation": "相同",
                    "description": "mock：与历史知识点相同。",
                }
            ],
            "expansion": [
                {"topic": "交叉验证", "description": "mock：历史知识库中未出现的新增知识点。", "is_new": True}
            ],
            "analysis": "mock：整体分析文本，用于验证导出排版。",
        }
    return {
        "overlap": [],
        "expansion": [
            {"topic": kp.get("topic"), "description": "首次运行，历史知识库为空，全部为新增。", "is_new": True}
            for kp in (new_summary or {}).get("knowledge_points") or []
        ],
        "analysis": "mock：历史知识库为空，本次总结全部为新增知识点。",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VidMind AI：总结 B站视频并对比历史知识库，导出 Word")
    p.add_argument("url", help="B站视频链接（--mock 时可传任意占位串）")
    p.add_argument("--kb", default=DEFAULT_KB, help="历史知识库 JSON 路径（默认 knowledge_base.json）")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="输出目录（默认 output）")
    p.add_argument("--cookies", default=None, help="可选：Netscape 格式 cookies.txt，用于需登录的字幕")
    p.add_argument("--mock", action="store_true", help="离线 dry-run：使用内置示例字幕与假总结，不联网不耗 API")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    load_dotenv()

    out_dir = os.path.abspath(args.out_dir)
    kb_path = os.path.abspath(args.kb)
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 1) 解析视频
        print("[1/6] 解析视频与字幕...")
        if args.mock:
            with open(SAMPLE_TRANSCRIPT, "r", encoding="utf-8") as f:
                transcript = f.read()
            title, bvid, url = "示例视频（mock）", "BV_mock", args.url
        else:
            info = parse_video(args.url, cookies_file=args.cookies)
            title, bvid, url, transcript = info.title, info.bvid, info.url, info.transcript
        print(f"      标题：{title}")
        print(f"      字幕长度：{len(transcript)} 字符")

        # 2) AI 总结
        print("[2/6] AI 总结并分类...")
        if args.mock:
            summary = _mock_summary(title)
        else:
            summary = DeepSeekClient().summarize(title, transcript)
        print(f"      知识点数量：{len(summary.get('knowledge_points') or [])}")

        # 3) 读历史知识库
        print("[3/6] 读取历史知识库...")
        old_kb = load_knowledge_base(kb_path)
        print(f"      历史视频数：{len(old_kb.get('videos') or [])}")

        # 4) AI 对比
        print("[4/6] AI 对比知识点...")
        if args.mock:
            comparison = _mock_compare(old_kb, summary)
        else:
            comparison = DeepSeekClient().compare(old_kb, summary)
        print(
            f"      重叠：{len(comparison.get('overlap') or [])} 条；"
            f"新增：{len(comparison.get('expansion') or [])} 条"
        )

        # 5) 导出 Word
        print("[5/6] 导出 Word...")
        meta = {
            "url": url,
            "bvid": bvid,
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        base = _safe_filename(title)
        docx_path = os.path.join(out_dir, base + ".docx")
        export_report(summary, comparison, meta, docx_path)
        print(f"      Word：{docx_path}")

        summary_json_path = os.path.join(out_dir, base + ".summary.json")
        comparison_json_path = os.path.join(out_dir, base + ".comparison.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(comparison_json_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)

        # 6) 保存知识库
        print("[6/6] 保存知识库...")
        entry = {
            "bvid": bvid,
            "title": title,
            "url": url,
            "summarized_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "summary": summary.get("summary") or "",
            "knowledge_points": summary.get("knowledge_points") or [],
        }
        _upsert_video(old_kb, entry)
        save_knowledge_base(kb_path, old_kb)
        print(f"      知识库：{kb_path}")

        print("\n[OK] 全部完成。")
        print(f"   Word      ：{docx_path}")
        print(f"   总结 JSON ：{summary_json_path}")
        print(f"   对比 JSON ：{comparison_json_path}")
        print(f"   知识库     ：{kb_path}")
        return 0
    except (VideoParseError, SummarizeError, ValueError) as e:
        print(f"\n[错误] {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"\n[文件缺失] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
